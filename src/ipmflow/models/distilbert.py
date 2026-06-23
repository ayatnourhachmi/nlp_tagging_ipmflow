"""Model D — DistilBERT with early stopping, class weights, and tuned multi-label thresholds."""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer, get_linear_schedule_with_warmup

from ipmflow.config import load_config
from ipmflow.data.load import load_dataset
from ipmflow.data.splits import ensure_split
from ipmflow.device import get_device
from ipmflow.eval.metrics import format_eval_result
from ipmflow.paths import MODEL_D_CHECKPOINTS, MODEL_D_EVAL, resolve
from ipmflow.taxonomy import (
    DOMAIN_CLASSES,
    IMPACT_CLASSES,
    MULTILABEL_THRESHOLD,
    OBJECTIVE_CLASSES,
    ORIGIN_CLASSES,
    confidence_band,
)


class PitchDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class DistilBERTClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float, bert_config=None):
        super().__init__()
        if bert_config is None:
            self.bert = DistilBertModel.from_pretrained(model_name)
        else:
            self.bert = DistilBertModel(bert_config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for block in self.bert.transformer.layer[:2]:
            for param in block.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_repr))


def _class_weights_from_onehot(y_onehot: np.ndarray, device: torch.device) -> torch.Tensor:
    y_idx = np.argmax(y_onehot, axis=1)
    classes = np.arange(y_onehot.shape[1])
    weights = compute_class_weight("balanced", classes=classes, y=y_idx)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _collect_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            logits = model(ids, mask)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def _predict_multilabel(logits: np.ndarray, thresholds: list[float] | None) -> np.ndarray:
    proba = 1.0 / (1.0 + np.exp(-logits))
    n_classes = proba.shape[1]
    if thresholds is None:
        thresholds = [MULTILABEL_THRESHOLD] * n_classes
    preds = np.zeros_like(proba, dtype=int)
    for j, threshold in enumerate(thresholds):
        preds[:, j] = (proba[:, j] >= threshold).astype(int)
    for i in range(len(preds)):
        if preds[i].sum() == 0:
            preds[i, int(np.argmax(proba[i]))] = 1
    return preds


def _score_multilabel(y_true: np.ndarray, logits: np.ndarray, thresholds: list[float] | None) -> float:
    preds = _predict_multilabel(logits, thresholds)
    return f1_score(y_true, preds, average="weighted", zero_division=0)


def _score_singlelabel(y_true_idx: np.ndarray, logits: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return f1_score(y_true_idx, preds, average="weighted", zero_division=0)


def tune_per_label_thresholds(y_true: np.ndarray, logits: np.ndarray, grid: np.ndarray | None = None) -> list[float]:
    proba = 1.0 / (1.0 + np.exp(-logits))
    if grid is None:
        grid = np.arange(0.15, 0.76, 0.05)
    thresholds = []
    for j in range(y_true.shape[1]):
        best_t, best_f1 = MULTILABEL_THRESHOLD, 0.0
        for threshold in grid:
            pred_j = (proba[:, j] >= threshold).astype(int)
            f1 = f1_score(y_true[:, j], pred_j, zero_division=0)
            if f1 >= best_f1:
                best_f1, best_t = f1, float(threshold)
        thresholds.append(best_t)
    return thresholds


def _train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total = 0.0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / len(loader)


def _make_val_split(idx_train: list[int], dataset: list[dict], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    stratify = [dataset[i]["labels"]["objective"] for i in idx_train]
    fit_idx, val_idx = train_test_split(
        idx_train,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify,
    )
    return fit_idx, val_idx


def _fit_classifier(
    name: str,
    texts: list[str],
    labels: np.ndarray,
    idx_train: list[int],
    idx_val: list[int],
    idx_test: list[int],
    class_names: list[str],
    is_multilabel: bool,
    cfg: dict,
    device: torch.device,
    use_class_weights: bool = False,
) -> dict:
    print(f"\n{'=' * 65}\n  Training BERT: {name.upper()} ({len(class_names)} classes)\n{'=' * 65}")

    tokenizer = DistilBertTokenizer.from_pretrained(cfg["model_name"])
    texts_train = [texts[i] for i in idx_train]
    texts_val = [texts[i] for i in idx_val]
    texts_test = [texts[i] for i in idx_test]
    y_train = labels[idx_train]
    y_val = labels[idx_val]
    y_test = labels[idx_test]

    train_ds = PitchDataset(texts_train, y_train, tokenizer, cfg["max_len"])
    val_ds = PitchDataset(texts_val, y_val, tokenizer, cfg["max_len"])
    test_ds = PitchDataset(texts_test, y_test, tokenizer, cfg["max_len"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"])
    test_dl = DataLoader(test_ds, batch_size=cfg["batch_size"])

    model = DistilBERTClassifier(cfg["model_name"], len(class_names), cfg["dropout"]).to(device)

    if is_multilabel:
        loss_fn = nn.BCEWithLogitsLoss()
        y_val_eval, y_test_eval = y_val, y_test
    else:
        class_weights = _class_weights_from_onehot(y_train, device) if use_class_weights else None
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        train_ds.labels = torch.argmax(train_ds.labels, dim=1).long()
        val_ds.labels = torch.argmax(val_ds.labels, dim=1).long()
        test_ds.labels = torch.argmax(test_ds.labels, dim=1).long()
        y_val_eval = np.argmax(y_val, axis=1)
        y_test_eval = np.argmax(y_test, axis=1)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    total_steps = len(train_dl) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg["warmup_ratio"]), total_steps
    )

    patience = cfg.get("early_stopping_patience", 4)
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    t0 = time.time()

    for epoch in range(cfg["epochs"]):
        tr_loss = _train_one_epoch(model, train_dl, optimizer, scheduler, loss_fn, device)
        val_logits, val_labels_tensor = _collect_logits(model, val_dl, device)
        if is_multilabel:
            val_f1 = _score_multilabel(val_labels_tensor, val_logits, None)
        else:
            val_f1 = _score_singlelabel(val_labels_tensor, val_logits)

        if epoch % 2 == 0 or epoch == cfg["epochs"] - 1:
            print(f"  Epoch {epoch:3d}: train_loss={tr_loss:.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"  Early stop at epoch {epoch} (best epoch {best_epoch}, val_f1={best_f1:.4f})")
                break

    train_s = time.time() - t0
    model.load_state_dict(best_state)

    val_logits, val_labels_arr = _collect_logits(model, val_dl, device)
    test_logits, test_labels_arr = _collect_logits(model, test_dl, device)

    thresholds = None
    threshold_map = None
    if is_multilabel:
        thresholds = tune_per_label_thresholds(val_labels_arr, val_logits)
        test_f1 = _score_multilabel(test_labels_arr, test_logits, thresholds)
        threshold_map = {class_names[i]: round(thresholds[i], 3) for i in range(len(class_names))}
        print(f"  Tuned thresholds: {threshold_map}")
    else:
        test_f1 = _score_singlelabel(y_test_eval, test_logits)

    save_path = MODEL_D_CHECKPOINTS[name]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "bert_config": model.bert.config,
        "num_classes": len(class_names),
        "is_multilabel": is_multilabel,
        "model_name": cfg["model_name"],
        "dropout": cfg["dropout"],
        "class_names": class_names,
        "thresholds": thresholds,
        "best_val_f1": round(float(best_f1), 4),
        "best_epoch": best_epoch,
        "train_config": {
            "lr": cfg["lr"],
            "epochs": cfg["epochs"],
            "early_stopping_patience": patience,
        },
    }
    torch.save(checkpoint, save_path)
    print(f"  Saved → {save_path.name}  |  test_f1={test_f1:.4f}  |  train={train_s:.1f}s")

    result = {
        "f1": round(float(test_f1), 4),
        "val_f1": round(float(best_f1), 4),
        "train_s": round(train_s, 1),
        "best_epoch": best_epoch,
    }
    if threshold_map:
        result["thresholds"] = threshold_map
    return result


def train(eval_path: Path | None = None) -> dict:
    cfg = load_config()["model_d"]
    device = get_device()
    dataset = load_dataset()
    split = ensure_split()
    idx_train, idx_test = split["idx_train"], split["idx_test"]
    idx_fit, idx_val = _make_val_split(idx_train, dataset, cfg.get("val_ratio", 0.15), split["random_state"])
    texts = [d["text"] for d in dataset]

    print(f"Dataset: {len(dataset)}  |  Fit: {len(idx_fit)}  |  Val: {len(idx_val)}  |  Test: {len(idx_test)}")
    print(f"Device: {device}  |  lr={cfg['lr']}  epochs={cfg['epochs']}  patience={cfg.get('early_stopping_patience', 4)}")

    y_obj = np.array(
        [[1 if d["labels"]["objective"] == c else 0 for c in OBJECTIVE_CLASSES] for d in dataset],
        dtype=np.float32,
    )
    mlb_dom = MultiLabelBinarizer(classes=DOMAIN_CLASSES)
    y_dom = mlb_dom.fit_transform([d["labels"]["domain"] for d in dataset]).astype(np.float32)
    mlb_imp = MultiLabelBinarizer(classes=IMPACT_CLASSES)
    y_imp = mlb_imp.fit_transform([d["labels"]["impact"] for d in dataset]).astype(np.float32)
    y_ori = np.array(
        [[1 if d["labels"]["origin"] == c else 0 for c in ORIGIN_CLASSES] for d in dataset],
        dtype=np.float32,
    )

    results = {
        "objective": _fit_classifier(
            "objective", texts, y_obj, idx_fit, idx_val, idx_test,
            OBJECTIVE_CLASSES, False, cfg, device, use_class_weights=True,
        ),
        "domain": _fit_classifier(
            "domain", texts, y_dom, idx_fit, idx_val, idx_test,
            DOMAIN_CLASSES, True, cfg, device,
        ),
        "impact": _fit_classifier(
            "impact", texts, y_imp, idx_fit, idx_val, idx_test,
            IMPACT_CLASSES, True, cfg, device,
        ),
        "origin": _fit_classifier(
            "origin", texts, y_ori, idx_fit, idx_val, idx_test,
            ORIGIN_CLASSES, False, cfg, device, use_class_weights=True,
        ),
    }

    avg_f1 = float(np.mean([results[k]["f1"] for k in results]))
    eval_out = format_eval_result(
        f"IPM Model D — DistilBERT ({cfg['model_name']}, lr={cfg['lr']}, early stopping)",
        {
            "objective": results["objective"]["f1"],
            "domain": results["domain"]["f1"],
            "impact": results["impact"]["f1"],
            "origin": results["origin"]["f1"],
            "avg_f1": round(avg_f1, 4),
        },
        len(idx_fit),
        len(idx_test),
        results,
    )
    eval_out["training"] = {
        "val_ratio": cfg.get("val_ratio", 0.15),
        "lr": cfg["lr"],
        "epochs": cfg["epochs"],
        "early_stopping_patience": cfg.get("early_stopping_patience", 4),
        "class_weights": ["objective", "origin"],
        "threshold_tuning": ["domain", "impact"],
    }

    out_eval = eval_path or MODEL_D_EVAL
    with open(out_eval, "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=2, ensure_ascii=False)
    print(f"\nAverage test F1: {avg_f1:.4f}  |  Saved eval → {out_eval}")
    return results


class _LoadedModels:
    def __init__(self):
        self.tokenizer = None
        self.models = {}
        self.thresholds = {}

    def load(self):
        if self.models:
            return
        cfg = load_config()["model_d"]
        self.tokenizer = DistilBertTokenizer.from_pretrained(cfg["model_name"])
        for name, path in MODEL_D_CHECKPOINTS.items():
            ckpt_path = resolve(path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run: python scripts/train.py d")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = DistilBERTClassifier(
                ckpt.get("model_name", cfg["model_name"]),
                ckpt["num_classes"],
                ckpt.get("dropout", cfg["dropout"]),
                bert_config=ckpt["bert_config"],
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.models[name] = model
            self.thresholds[name] = ckpt.get("thresholds")


_CACHE = _LoadedModels()


def _multilabel_tags(proba: np.ndarray, class_names: list[str], thresholds: list[float] | None) -> list[dict]:
    if thresholds is None:
        thresholds = [MULTILABEL_THRESHOLD] * len(class_names)
    mask = np.array([proba[i] >= thresholds[i] for i in range(len(class_names))])
    if not mask.any():
        mask[int(np.argmax(proba))] = True
    return [
        {"value": class_names[i], "confidence": confidence_band(float(proba[i]))}
        for i in range(len(class_names))
        if mask[i]
    ]


def predict(text: str) -> dict:
    _CACHE.load()
    cfg = load_config()["model_d"]
    enc = _CACHE.tokenizer(
        text, truncation=True, padding=True, max_length=cfg["max_len"], return_tensors="pt"
    )
    ids, mask = enc["input_ids"], enc["attention_mask"]
    t0 = time.time()

    with torch.no_grad():
        obj_logits = _CACHE.models["objective"](ids, mask)
        obj_proba = torch.softmax(obj_logits, dim=1).numpy()[0]
        dom_proba = torch.sigmoid(_CACHE.models["domain"](ids, mask)).numpy()[0]
        imp_proba = torch.sigmoid(_CACHE.models["impact"](ids, mask)).numpy()[0]
        ori_logits = _CACHE.models["origin"](ids, mask)
        ori_proba = torch.softmax(ori_logits, dim=1).numpy()[0]

    return {
        "tags": {
            "objective": {
                "value": OBJECTIVE_CLASSES[int(np.argmax(obj_proba))],
                "confidence": confidence_band(float(obj_proba.max())),
            },
            "domain": _multilabel_tags(dom_proba, DOMAIN_CLASSES, _CACHE.thresholds.get("domain")),
            "impact": _multilabel_tags(imp_proba, IMPACT_CLASSES, _CACHE.thresholds.get("impact")),
            "origin": {
                "value": ORIGIN_CLASSES[int(np.argmax(ori_proba))],
                "confidence": confidence_band(float(ori_proba.max())),
            },
        },
        "_meta": {"model": "model_d_distilbert", "latency_ms": round((time.time() - t0) * 1000, 2)},
    }


def extract_labels(result: dict) -> dict:
    tags = result["tags"]
    return {
        "objective": tags["objective"]["value"],
        "domain": [d["value"] for d in tags["domain"]],
        "impact": [d["value"] for d in tags["impact"]],
        "origin": tags["origin"]["value"],
    }
