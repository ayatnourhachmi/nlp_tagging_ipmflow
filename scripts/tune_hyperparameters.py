#!/usr/bin/env python
"""Tune hyperparameters for IPM Flow models."""

import argparse
import copy
import itertools
import json
import sys

import _bootstrap  # noqa: F401

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup

from ipmflow.config import load_config
from ipmflow.data.generate import TARGET_N, build_dataset, print_stats
from ipmflow.data.splits import create_split
from ipmflow.device import get_device
from ipmflow.models.distilbert import (
    DistilBERTClassifier,
    PitchDataset,
    _class_weights_from_onehot,
    _collect_logits,
    _score_multilabel,
    _score_singlelabel,
    _train_one_epoch,
    tune_per_label_thresholds,
)
from ipmflow.models.sklearn_tfidf import model_c_logreg_kwargs, model_c_tfidf_kwargs
from ipmflow.paths import DATASET_PATH
from ipmflow.taxonomy import (
    DOMAIN_CLASSES,
    IMPACT_CLASSES,
    OBJECTIVE_CLASSES,
    ORIGIN_CLASSES,
)

MODEL_D_HEADS = {
    "objective": {
        "classes": OBJECTIVE_CLASSES,
        "multilabel": False,
        "class_weights": True,
    },
    "origin": {
        "classes": ORIGIN_CLASSES,
        "multilabel": False,
        "class_weights": True,
    },
    "domain": {
        "classes": DOMAIN_CLASSES,
        "multilabel": True,
        "class_weights": False,
    },
    "impact": {
        "classes": IMPACT_CLASSES,
        "multilabel": True,
        "class_weights": False,
    },
}

MODEL_D_PARAM_GRID = {
    "lr": [5e-6, 1e-5, 2e-5],
    "dropout": [0.2, 0.3, 0.4],
    "weight_decay": [0.0, 0.01],
}


def set_seed(seed: int) -> None:
    """Fix RNG state for reproducible tuning runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataset_and_split():
    """Generate dataset, save split, and return indices for tuning."""
    print("Generating and splitting data...")
    dataset = build_dataset(TARGET_N)
    print_stats(dataset)

    split_info = create_split(dataset)

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nSaved dataset to {DATASET_PATH}")
    print(f"Data split: {split_info['n_train']} train / {split_info['n_test']} test")

    idx_train, idx_test = split_info["idx_train"], split_info["idx_test"]
    if not idx_train:
        print("Warning: No training data found for train indices.")
        print("Dataset structure might be unexpected or split might be empty.")

    return dataset, idx_train, idx_test


def generate_and_split_data():
    """Return train/test text + objective labels (Model C and legacy Model D API)."""
    dataset, idx_train, idx_test = prepare_dataset_and_split()
    X_train = [dataset[i]["text"] for i in idx_train]
    y_train = [dataset[i]["labels"]["objective"] for i in idx_train]
    X_test = [dataset[i]["text"] for i in idx_test]
    y_test = [dataset[i]["labels"]["objective"] for i in idx_test]
    return X_train, y_train, X_test, y_test


def build_head_tune_arrays(dataset, idx_train, idx_test, head: str):
    """Build aligned texts, one-hot/multi-hot labels, and stratify keys for a head."""
    spec = MODEL_D_HEADS[head]
    texts = [dataset[i]["text"] for i in idx_train + idx_test]
    n_train = len(idx_train)

    if spec["multilabel"]:
        from sklearn.preprocessing import MultiLabelBinarizer

        mlb = MultiLabelBinarizer(classes=spec["classes"])
        labels = mlb.fit_transform(
            [dataset[i]["labels"][head] for i in idx_train + idx_test]
        ).astype(np.float32)
        stratify = [dataset[i]["labels"]["objective"] for i in idx_train]
    else:
        labels = np.array(
            [
                [1 if dataset[i]["labels"][head] == c else 0 for c in spec["classes"]]
                for i in idx_train + idx_test
            ],
            dtype=np.float32,
        )
        stratify = [dataset[i]["labels"][head] for i in idx_train]

    return texts, labels, stratify, n_train, spec


def tune_model_c(X_train, y_train, X_test, y_test):
    """Tune Model C hyperparameters on objective using the production vectorizer settings."""
    cfg = load_config()["model_c"]
    n_classes = len(set(y_train))
    print("\nStarting hyperparameter tuning for Model C (TF-IDF + Logistic Regression)...")
    print(f"Tuning on 'objective' ({n_classes} classes, {len(X_train)} train / {len(X_test)} test)")
    print(f"Base vectorizer settings: {model_c_tfidf_kwargs(cfg)}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**model_c_tfidf_kwargs(cfg))),
        (
            "clf",
            LogisticRegression(
                solver="saga",
                **{k: v for k, v in model_c_logreg_kwargs(cfg).items() if k != "C"},
            ),
        ),
    ])

    param_grid = {
        "tfidf__max_features": [5000, 8000, 10000, 15000],
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "clf__C": [0.5, 1.0, 2.0, 5.0, 10.0],
        "clf__l1_ratio": [0.0, 1.0],
    }

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    best = grid_search.best_params_
    y_pred = grid_search.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n--- Hyperparameter Tuning Results for Model C ---")
    print(f"Best parameters found: {best}")
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    print(f"Held-out test F1 (objective): {test_f1:.4f}")
    print("\nSuggested config/model_c.yaml snippet:")
    print("model_c:")
    print(f"  max_features: {best['tfidf__max_features']}")
    print(f"  C: {best['clf__C']}")
    print(f"  max_iter: {cfg['max_iter']}")
    print(f"  ngram_range: [{best['tfidf__ngram_range'][0]}, {best['tfidf__ngram_range'][1]}]")
    print(f"  sublinear_tf: {cfg.get('sublinear_tf', True)}")
    print(f"  stop_words: {cfg.get('stop_words', 'english')}")
    print(f"  min_df: {cfg.get('min_df', 1)}")
    if best["clf__l1_ratio"] == 1.0:
        print("  # Note: best CV run used L1; production LogisticRegression defaults to L2 (lbfgs).")


def _fit_head_trial(
    texts: list[str],
    labels: np.ndarray,
    idx_fit: list[int],
    idx_val: list[int],
    idx_test: list[int],
    cfg: dict,
    device: torch.device,
    tokenizer: DistilBertTokenizer,
    head_spec: dict,
    seed: int,
) -> dict:
    """Train one classifier head from scratch for a single hyperparameter combo."""
    class_names = head_spec["classes"]
    is_multilabel = head_spec["multilabel"]
    use_class_weights = head_spec["class_weights"]

    texts_fit = [texts[i] for i in idx_fit]
    texts_val = [texts[i] for i in idx_val]
    texts_test = [texts[i] for i in idx_test]
    y_fit = labels[idx_fit]
    y_val = labels[idx_val]
    y_test = labels[idx_test]

    train_ds = PitchDataset(texts_fit, y_fit, tokenizer, cfg["max_len"])
    val_ds = PitchDataset(texts_val, y_val, tokenizer, cfg["max_len"])
    test_ds = PitchDataset(texts_test, y_test, tokenizer, cfg["max_len"])
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_dl = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=generator
    )
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"])
    test_dl = DataLoader(test_ds, batch_size=cfg["batch_size"])

    model = DistilBERTClassifier(cfg["model_name"], len(class_names), cfg["dropout"]).to(device)

    if is_multilabel:
        loss_fn = nn.BCEWithLogitsLoss()
        y_val_eval, y_test_eval = y_val, y_test
    else:
        class_weights = _class_weights_from_onehot(y_fit, device) if use_class_weights else None
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        for ds in (train_ds, val_ds, test_ds):
            ds.labels = torch.argmax(ds.labels, dim=1).long()
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

    for epoch in range(cfg["epochs"]):
        _train_one_epoch(model, train_dl, optimizer, scheduler, loss_fn, device)
        val_logits, val_labels_tensor = _collect_logits(model, val_dl, device)
        if is_multilabel:
            val_f1 = _score_multilabel(val_labels_tensor, val_logits, None)
        else:
            val_f1 = _score_singlelabel(val_labels_tensor, val_logits)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(
                    f"  Early stop at epoch {epoch} "
                    f"(best epoch {best_epoch}, val_f1={best_f1:.4f})"
                )
                break

    model.load_state_dict(best_state)
    val_logits, val_labels_arr = _collect_logits(model, val_dl, device)
    test_logits, test_labels_arr = _collect_logits(model, test_dl, device)

    thresholds = None
    if is_multilabel:
        thresholds = tune_per_label_thresholds(val_labels_arr, val_logits)
        test_f1 = _score_multilabel(test_labels_arr, test_logits, thresholds)
    else:
        test_f1 = _score_singlelabel(y_test_eval, test_logits)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    result = {"val_f1": float(best_f1), "test_f1": float(test_f1), "best_epoch": best_epoch}
    if thresholds is not None:
        result["thresholds"] = {
            class_names[i]: round(thresholds[i], 3) for i in range(len(class_names))
        }
    return result


def _print_model_d_results(best, results, cfg_base, head: str) -> None:
    best_params = best["params"]
    print(f"\n--- Hyperparameter Tuning Results for Model D ({head}) ---")
    print(f"Best parameters found (by val_f1): {best_params}")
    print(f"Best validation F1 ({head}): {best['val_f1']:.4f}")
    print(f"Held-out test F1 at best val combo ({head}): {best['test_f1']:.4f}")
    if best.get("thresholds"):
        print(f"Tuned thresholds: {best['thresholds']}")
    print("\nSuggested config/model_d.yaml snippet (shared across heads):")
    print("model_d:")
    print(f"  model_name: {cfg_base['model_name']}")
    print(f"  max_len: {cfg_base['max_len']}")
    print(f"  batch_size: {cfg_base['batch_size']}")
    print(f"  epochs: {cfg_base['epochs']}")
    print(f"  lr: {best_params['lr']}")
    print(f"  warmup_ratio: {cfg_base['warmup_ratio']}")
    print(f"  dropout: {best_params['dropout']}")
    print(f"  weight_decay: {best_params['weight_decay']}")
    print(f"  val_ratio: {cfg_base.get('val_ratio', 0.15)}")
    print(f"  early_stopping_patience: {cfg_base.get('early_stopping_patience', 4)}")
    if head != "objective":
        print(f"  # Tuned on '{head}' — re-check other heads after updating shared model_d settings.")


def tune_model_d(
    dataset,
    idx_train,
    idx_test,
    head: str = "objective",
    seed: int | None = None,
    single: bool = False,
):
    """Tune Model D on one head via grid search, or one reproducible run with --single."""
    if head not in MODEL_D_HEADS:
        raise ValueError(f"Unknown head {head!r}; choose from {list(MODEL_D_HEADS)}")

    cfg = load_config()
    cfg_base = cfg["model_d"]
    split_cfg = cfg.get("split", {})
    seed = seed if seed is not None else split_cfg.get("random_state", 42)
    set_seed(seed)

    head_spec = MODEL_D_HEADS[head]
    device = get_device()
    texts, labels, stratify, n_train, _ = build_head_tune_arrays(
        dataset, idx_train, idx_test, head
    )
    n_test = len(idx_test)

    print(f"\nStarting hyperparameter tuning for Model D (DistilBERT) — head: {head}")
    print(
        f"Tuning on '{head}' ({len(head_spec['classes'])} classes, "
        f"{n_train} train / {n_test} test)"
    )
    print(f"Device: {device}  |  seed={seed}  |  Base lr={cfg_base['lr']}, dropout={cfg_base['dropout']}")
    if head_spec["multilabel"]:
        print("Loss: BCEWithLogits + per-label threshold tuning on val")
    elif head_spec["class_weights"]:
        print("Loss: weighted CrossEntropy (balanced class weights)")
    else:
        print("Loss: CrossEntropy")

    idx_all = list(range(n_train))
    idx_fit, idx_val = train_test_split(
        idx_all,
        test_size=cfg_base.get("val_ratio", 0.15),
        random_state=seed,
        stratify=stratify,
    )
    idx_test_local = list(range(n_train, len(texts)))

    if single:
        combos = [(cfg_base["lr"], cfg_base["dropout"], cfg_base.get("weight_decay", 0.01))]
        print("Mode: single run (current config, fixed seed — no grid search)")
    else:
        combos = list(itertools.product(*(MODEL_D_PARAM_GRID[k] for k in MODEL_D_PARAM_GRID)))
        print(f"Exhaustive search: {len(combos)} combinations via itertools.product")
        print(f"  lr: {MODEL_D_PARAM_GRID['lr']}")
        print(f"  dropout: {MODEL_D_PARAM_GRID['dropout']}")
        print(f"  weight_decay: {MODEL_D_PARAM_GRID['weight_decay']}")

    print("Selection metric: validation F1 (weighted)")
    print(f"Val split: {len(idx_fit)} fit / {len(idx_val)} val / {len(idx_test_local)} held-out test")

    tokenizer = DistilBertTokenizer.from_pretrained(cfg_base["model_name"])
    results = []

    for i, (lr, dropout, weight_decay) in enumerate(combos, start=1):
        trial_cfg = {**cfg_base, "lr": lr, "dropout": dropout, "weight_decay": weight_decay}
        print(f"\n[{i}/{len(combos)}] lr={lr:g}, dropout={dropout}, weight_decay={weight_decay}")
        trial = _fit_head_trial(
            texts,
            labels,
            idx_fit,
            idx_val,
            idx_test_local,
            trial_cfg,
            device,
            tokenizer,
            head_spec,
            seed,
        )
        trial["params"] = {"lr": lr, "dropout": dropout, "weight_decay": weight_decay}
        results.append(trial)
        extra = ""
        if trial.get("thresholds"):
            extra = f"  thresholds={trial['thresholds']}"
        print(
            f"  val_f1={trial['val_f1']:.4f}  test_f1={trial['test_f1']:.4f}  "
            f"best_epoch={trial['best_epoch']}{extra}"
        )

    if single:
        trial = results[0]
        print(f"\n--- Single run result (Model D / {head}) ---")
        print(f"  lr={trial['params']['lr']:g}  dropout={trial['params']['dropout']}  "
              f"weight_decay={trial['params']['weight_decay']}")
        print(f"  val_f1={trial['val_f1']:.4f}  test_f1={trial['test_f1']:.4f}  "
              f"best_epoch={trial['best_epoch']}")
        if trial.get("thresholds"):
            print(f"  thresholds={trial['thresholds']}")
        return

    best = max(results, key=lambda r: r["val_f1"])

    print("\n--- All trials (ranked by val_f1) ---")
    for rank, trial in enumerate(sorted(results, key=lambda r: r["val_f1"], reverse=True), start=1):
        p = trial["params"]
        marker = " <-- best" if trial is best else ""
        print(
            f"  {rank:2d}. lr={p['lr']:g}  dropout={p['dropout']}  "
            f"weight_decay={p['weight_decay']}  "
            f"val_f1={trial['val_f1']:.4f}  test_f1={trial['test_f1']:.4f}  "
            f"epoch={trial['best_epoch']}{marker}"
        )

    _print_model_d_results(best, results, cfg_base, head)


def main():
    parser = argparse.ArgumentParser(description="Tune IPM Flow models")
    parser.add_argument(
        "model",
        choices=["c", "d"],
        help="Model to tune: c (sklearn_tfidf) or d (distilbert)",
    )
    parser.add_argument(
        "--head",
        choices=list(MODEL_D_HEADS),
        default="objective",
        help="Model D classifier head to tune (default: objective)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible trials (default: split.random_state from config)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Model D only: one run with current config (no grid search)",
    )
    args = parser.parse_args()

    if args.model == "c":
        try:
            X_train, y_train, X_test, y_test = generate_and_split_data()
        except (FileNotFoundError, KeyError) as e:
            print(f"Error preparing data: {e}")
            sys.exit(1)
        if not X_train or not y_train:
            print("Cannot proceed with tuning as training data is empty.")
            return
        tune_model_c(X_train, y_train, X_test, y_test)
        return

    try:
        dataset, idx_train, idx_test = prepare_dataset_and_split()
    except (FileNotFoundError, KeyError) as e:
        print(f"Error preparing data: {e}")
        sys.exit(1)

    if not idx_train:
        print("Cannot proceed with tuning as training data is empty.")
        return

    tune_model_d(
        dataset,
        idx_train,
        idx_test,
        head=args.head,
        seed=args.seed,
        single=args.single,
    )


if __name__ == "__main__":
    main()
