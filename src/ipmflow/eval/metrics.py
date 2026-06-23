"""Evaluation metrics for IPM Flow classifiers."""

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from ipmflow.taxonomy import DOMAIN_CLASSES, IMPACT_CLASSES


def compute_dimension_f1(ground_truth: list[dict], predictions: list[dict]) -> dict[str, float]:
    y_obj_true = [g["objective"] for g in ground_truth]
    y_obj_pred = [p["objective"] for p in predictions]
    f1_objective = f1_score(y_obj_true, y_obj_pred, average="weighted", zero_division=0)

    y_ori_true = [g["origin"] for g in ground_truth]
    y_ori_pred = [p["origin"] for p in predictions]
    f1_origin = f1_score(y_ori_true, y_ori_pred, average="weighted", zero_division=0)

    mlb_dom = MultiLabelBinarizer(classes=DOMAIN_CLASSES)
    y_dom_true = mlb_dom.fit_transform([g["domain"] for g in ground_truth])
    y_dom_pred = mlb_dom.transform([p["domain"] for p in predictions])
    f1_domain = f1_score(y_dom_true, y_dom_pred, average="weighted", zero_division=0)

    mlb_imp = MultiLabelBinarizer(classes=IMPACT_CLASSES)
    y_imp_true = mlb_imp.fit_transform([g["impact"] for g in ground_truth])
    y_imp_pred = mlb_imp.transform([p["impact"] for p in predictions])
    f1_impact = f1_score(y_imp_true, y_imp_pred, average="weighted", zero_division=0)

    scores = {
        "objective": round(f1_objective, 4),
        "domain": round(f1_domain, 4),
        "impact": round(f1_impact, 4),
        "origin": round(f1_origin, 4),
    }
    scores["avg_f1"] = round(sum(scores.values()) / 4, 4)
    return scores


def format_eval_result(
    model_name: str,
    scores: dict[str, float],
    n_train: int,
    n_test: int,
    classifiers: dict | None = None,
) -> dict:
    return {
        "model": model_name,
        "avg_f1": scores["avg_f1"],
        "n_train": n_train,
        "n_test": n_test,
        "classifiers": classifiers
        or {k: {"f1": scores[k]} for k in ("objective", "domain", "impact", "origin")},
    }
