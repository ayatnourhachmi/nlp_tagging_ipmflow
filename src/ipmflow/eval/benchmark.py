"""Unified benchmark across LLM and trained models."""

import json
import re
import time
from pathlib import Path

from ipmflow.data.load import load_dataset
from ipmflow.data.splits import ensure_split
from ipmflow.eval.metrics import compute_dimension_f1
from ipmflow.llm.providers import PROVIDERS
from ipmflow.models import distilbert, sklearn_tfidf


def parse_llm_json(raw: str) -> dict:
    cleaned = re.sub(r"```json|```", "", raw).strip()
    return json.loads(cleaned)


def extract_llm_labels(parsed: dict) -> dict:
    tags = parsed.get("tags", {})
    return {
        "objective": tags.get("objective", {}).get("value"),
        "domain": [d["value"] for d in tags.get("domain", [])],
        "impact": [d["value"] for d in tags.get("impact", [])],
        "origin": tags.get("origin", {}).get("value"),
    }


def evaluate_model(model_name: str, pitches: list, ground_truth: list, infer_fn) -> dict:
    predictions = []
    latencies = []
    errors = []

    print(f"\n{'=' * 65}\nEvaluating {model_name} on {len(pitches)} pitches\n{'=' * 65}")

    for i, text in enumerate(pitches):
        t0 = time.time()
        try:
            pred = infer_fn(text)
            latencies.append((time.time() - t0) * 1000)
            predictions.append(pred)
            status = "OK"
        except Exception as exc:
            errors.append({"index": i, "text": text[:80], "error": str(exc)})
            predictions.append({"objective": None, "domain": [], "impact": [], "origin": None})
            status = f"ERR: {exc}"
        print(f"  [{i + 1}/{len(pitches)}] {status}")

    valid_idx = [i for i, p in enumerate(predictions) if p["objective"] is not None]
    valid_pred = [predictions[i] for i in valid_idx]
    valid_gt = [ground_truth[i] for i in valid_idx]
    scores = compute_dimension_f1(valid_gt, valid_pred)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    result = {
        "model": model_name,
        "avg_f1": scores["avg_f1"],
        "n_test": len(pitches),
        "n_success": len(valid_pred),
        "n_errors": len(errors),
        "avg_latency_ms": round(avg_latency, 1),
        "classifiers": {k: {"f1": scores[k]} for k in ("objective", "domain", "impact", "origin")},
        "errors": errors,
    }

    for dim in ("objective", "domain", "impact", "origin"):
        print(f"  {dim:10s} F1: {scores[dim]:.4f}")
    print(f"  avg F1: {scores['avg_f1']:.4f}  |  latency: {avg_latency:.0f}ms")
    return result


def get_test_set(limit: int = 0):
    dataset = load_dataset()
    split = ensure_split()
    idx_test = split["idx_test"]
    pitches = [dataset[i]["text"] for i in idx_test]
    ground_truth = [dataset[i]["labels"] for i in idx_test]
    if limit > 0:
        pitches = pitches[:limit]
        ground_truth = ground_truth[:limit]
    return pitches, ground_truth


def benchmark_llm(provider: str, limit: int = 0) -> dict:
    cfg = PROVIDERS[provider]
    pitches, ground_truth = get_test_set(limit)

    def infer(text):
        raw = cfg["fn"](text)
        return extract_llm_labels(parse_llm_json(raw))

    return evaluate_model(f"IPM {cfg['label']} — {cfg['model']}", pitches, ground_truth, infer)


def benchmark_trained(models: list[str], limit: int = 0) -> list[dict]:
    pitches, ground_truth = get_test_set(limit)
    results = []

    if "c" in models:
        artifact_c = sklearn_tfidf.load_artifact()
        results.append(
            evaluate_model(
                "IPM Model C — TF-IDF + LogisticRegression",
                pitches,
                ground_truth,
                lambda t: sklearn_tfidf.extract_labels(sklearn_tfidf.predict(t, artifact_c)),
            )
        )

    if "d" in models:
        results.append(
            evaluate_model(
                "IPM Model D — DistilBERT (early stopping, tuned thresholds)",
                pitches,
                ground_truth,
                lambda t: distilbert.extract_labels(distilbert.predict(t)),
            )
        )

    return results


def benchmark_all(limit: int = 0) -> dict:
    results = []
    for provider in ("groq", "openai"):
        try:
            results.append(benchmark_llm(provider, limit))
        except Exception as exc:
            print(f"Skipping {provider}: {exc}")
    results.extend(benchmark_trained(["c", "d"], limit))
    return {"benchmark_date": time.strftime("%Y-%m-%d %H:%M"), "n_test": len(get_test_set(limit)[0]), "models": results}


def save_result(result: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")
