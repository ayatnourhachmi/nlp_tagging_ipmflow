"""Model C — TF-IDF + Logistic Regression."""

import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from ipmflow.config import load_config
from ipmflow.data.load import load_dataset
from ipmflow.data.splits import ensure_split
from ipmflow.eval.metrics import format_eval_result
from ipmflow.paths import MODEL_C_ARTIFACT, MODEL_C_EVAL, resolve
from ipmflow.taxonomy import (
    DOMAIN_CLASSES,
    IMPACT_CLASSES,
    MULTILABEL_THRESHOLD,
    confidence_band,
)


def _ngram_range(cfg: dict) -> tuple[int, int]:
    ngram_range = cfg.get("ngram_range", [1, 3])
    return tuple(ngram_range)


def model_c_tfidf_kwargs(cfg: dict | None = None, **overrides) -> dict:
    """TF-IDF settings shared by training and hyperparameter tuning."""
    merged = {**load_config()["model_c"], **(cfg or {}), **overrides}
    return {
        "ngram_range": _ngram_range(merged),
        "max_features": merged["max_features"],
        "sublinear_tf": merged.get("sublinear_tf", True),
        "analyzer": "word",
        "stop_words": merged.get("stop_words", "english"),
        "min_df": merged.get("min_df", 1),
    }


def model_c_logreg_kwargs(cfg: dict | None = None, **overrides) -> dict:
    """LogisticRegression settings shared by training and hyperparameter tuning."""
    merged = {**load_config()["model_c"], **(cfg or {}), **overrides}
    return {
        "max_iter": merged["max_iter"],
        "C": merged["C"],
        "random_state": 42,
        "class_weight": "balanced",
    }


def train(artifact_path: Path | None = None, eval_path: Path | None = None) -> dict:
    cfg = load_config()["model_c"]
    dataset = load_dataset()
    split = ensure_split()
    idx_train, idx_test = split["idx_train"], split["idx_test"]

    texts = [d["text"] for d in dataset]
    vectorizer = TfidfVectorizer(**model_c_tfidf_kwargs(cfg))
    X = vectorizer.fit_transform(texts)
    X_train, X_test = X[idx_train], X[idx_test]

    print("=" * 65)
    print("MODEL C — TF-IDF + Logistic Regression")
    print(f"Train: {len(idx_train)} | Test: {len(idx_test)}")
    print("=" * 65)

    results = {}
    models = {}

    y_objective = [d["labels"]["objective"] for d in dataset]
    y_obj_train = [y_objective[i] for i in idx_train]
    y_obj_test = [y_objective[i] for i in idx_test]

    clf_objective = LogisticRegression(**model_c_logreg_kwargs(cfg))
    t0 = time.time()
    clf_objective.fit(X_train, y_obj_train)
    train_t = time.time() - t0
    t0 = time.time()
    y_pred_obj = clf_objective.predict(X_test)
    infer_t = (time.time() - t0) / len(idx_test) * 1000
    f1 = f1_score(y_obj_test, y_pred_obj, average="weighted", zero_division=0)
    models["objective"] = clf_objective
    results["objective"] = {
        "f1": round(f1, 4),
        "train_s": round(train_t, 3),
        "infer_ms": round(infer_t, 3),
        "classes": clf_objective.classes_.tolist(),
    }
    print(f"\n[OBJECTIVE] F1={f1:.4f}")
    print(classification_report(y_obj_test, y_pred_obj, zero_division=0))

    mlb_domain = MultiLabelBinarizer(classes=DOMAIN_CLASSES)
    y_domain = mlb_domain.fit_transform([d["labels"]["domain"] for d in dataset])
    y_dom_train, y_dom_test = y_domain[idx_train], y_domain[idx_test]
    clf_domain = OneVsRestClassifier(LogisticRegression(**model_c_logreg_kwargs(cfg)))
    t0 = time.time()
    clf_domain.fit(X_train, y_dom_train)
    train_t = time.time() - t0
    t0 = time.time()
    y_pred_dom = clf_domain.predict(X_test)
    infer_t = (time.time() - t0) / len(idx_test) * 1000
    f1 = f1_score(y_dom_test, y_pred_dom, average="weighted", zero_division=0)
    models["domain"] = clf_domain
    results["domain"] = {"f1": round(f1, 4), "train_s": round(train_t, 3), "infer_ms": round(infer_t, 3)}

    mlb_impact = MultiLabelBinarizer(classes=IMPACT_CLASSES)
    y_impact = mlb_impact.fit_transform([d["labels"]["impact"] for d in dataset])
    y_imp_train, y_imp_test = y_impact[idx_train], y_impact[idx_test]
    clf_impact = OneVsRestClassifier(LogisticRegression(**model_c_logreg_kwargs(cfg)))
    t0 = time.time()
    clf_impact.fit(X_train, y_imp_train)
    train_t = time.time() - t0
    t0 = time.time()
    y_pred_imp = clf_impact.predict(X_test)
    infer_t = (time.time() - t0) / len(idx_test) * 1000
    f1 = f1_score(y_imp_test, y_pred_imp, average="weighted", zero_division=0)
    models["impact"] = clf_impact
    results["impact"] = {"f1": round(f1, 4), "train_s": round(train_t, 3), "infer_ms": round(infer_t, 3)}

    y_origin = [d["labels"]["origin"] for d in dataset]
    y_ori_train = [y_origin[i] for i in idx_train]
    y_ori_test = [y_origin[i] for i in idx_test]
    clf_origin = LogisticRegression(**model_c_logreg_kwargs(cfg))
    t0 = time.time()
    clf_origin.fit(X_train, y_ori_train)
    train_t = time.time() - t0
    t0 = time.time()
    y_pred_ori = clf_origin.predict(X_test)
    infer_t = (time.time() - t0) / len(idx_test) * 1000
    f1 = f1_score(y_ori_test, y_pred_ori, average="weighted", zero_division=0)
    models["origin"] = clf_origin
    results["origin"] = {
        "f1": round(f1, 4),
        "train_s": round(train_t, 3),
        "infer_ms": round(infer_t, 3),
        "classes": clf_origin.classes_.tolist(),
    }

    avg_f1 = float(np.mean([results[k]["f1"] for k in results]))
    print(f"\nAverage F1: {avg_f1:.4f}")

    artifact = {
        "vectorizer": vectorizer,
        "models": models,
        "mlb_domain": mlb_domain,
        "mlb_impact": mlb_impact,
        "results": results,
        "avg_f1": round(avg_f1, 4),
        "idx_train": idx_train,
        "idx_test": idx_test,
    }

    out_artifact = artifact_path or MODEL_C_ARTIFACT
    out_artifact.parent.mkdir(parents=True, exist_ok=True)
    with open(out_artifact, "wb") as f:
        pickle.dump(artifact, f)

    eval_out = format_eval_result(
        "IPM Model C — TF-IDF trigram + LogisticRegression (OVR for multi-label)",
        {"objective": results["objective"]["f1"], "domain": results["domain"]["f1"],
         "impact": results["impact"]["f1"], "origin": results["origin"]["f1"],
         "avg_f1": round(avg_f1, 4)},
        len(idx_train),
        len(idx_test),
        results,
    )
    out_eval = eval_path or MODEL_C_EVAL
    with open(out_eval, "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out_artifact}  |  {out_eval}")
    return artifact


def load_artifact(path: Path | None = None) -> dict:
    artifact_path = resolve(path or MODEL_C_ARTIFACT)
    with open(artifact_path, "rb") as f:
        return pickle.load(f)


def predict(text: str, artifact: dict) -> dict:
    vec = artifact["vectorizer"]
    models = artifact["models"]
    mlb_dom = artifact["mlb_domain"]
    mlb_imp = artifact["mlb_impact"]

    X = vec.transform([text])
    t0 = time.time()

    obj_proba = models["objective"].predict_proba(X)[0]
    obj_idx = int(np.argmax(obj_proba))
    obj_val = models["objective"].classes_[obj_idx]
    obj_conf = float(obj_proba[obj_idx])

    dom_probas = np.array([est.predict_proba(X)[0][1] for est in models["domain"].estimators_])
    dom_mask = dom_probas >= MULTILABEL_THRESHOLD
    if not dom_mask.any():
        dom_mask[np.argmax(dom_probas)] = True
    dom_classes = mlb_dom.classes_
    domain_out = [
        {"value": dom_classes[i], "confidence": confidence_band(dom_probas[i])}
        for i in range(len(dom_classes))
        if dom_mask[i]
    ]

    imp_probas = np.array([est.predict_proba(X)[0][1] for est in models["impact"].estimators_])
    imp_mask = imp_probas >= MULTILABEL_THRESHOLD
    if not imp_mask.any():
        imp_mask[np.argmax(imp_probas)] = True
    imp_classes = mlb_imp.classes_
    impact_out = [
        {"value": imp_classes[i], "confidence": confidence_band(imp_probas[i])}
        for i in range(len(imp_classes))
        if imp_mask[i]
    ]

    ori_proba = models["origin"].predict_proba(X)[0]
    ori_idx = int(np.argmax(ori_proba))
    ori_val = models["origin"].classes_[ori_idx]
    ori_conf = float(ori_proba[ori_idx])

    return {
        "tags": {
            "objective": {"value": obj_val, "confidence": confidence_band(obj_conf)},
            "domain": domain_out,
            "impact": impact_out,
            "origin": {"value": ori_val, "confidence": confidence_band(ori_conf)},
        },
        "_meta": {
            "model": "model_c_tfidf_lr",
            "latency_ms": round((time.time() - t0) * 1000, 2),
        },
    }


def extract_labels(result: dict) -> dict:
    tags = result["tags"]
    return {
        "objective": tags["objective"]["value"],
        "domain": [d["value"] for d in tags["domain"]],
        "impact": [d["value"] for d in tags["impact"]],
        "origin": tags["origin"]["value"],
    }
