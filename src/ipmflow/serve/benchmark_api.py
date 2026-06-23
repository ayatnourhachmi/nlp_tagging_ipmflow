"""Load benchmark eval JSON for the dashboard."""

import json

from ipmflow.paths import GROQ_EVAL, MODEL_C_EVAL, MODEL_D_EVAL, OPENAI_EVAL, resolve

BENCHMARK_SOURCES = (
    ("a", "Model A — Groq LLM", GROQ_EVAL),
    ("b", "Model B — OpenAI LLM", OPENAI_EVAL),
    ("c", "Model C — TF-IDF + LR", MODEL_C_EVAL),
    ("d", "Model D — DistilBERT", MODEL_D_EVAL),
)


def load_benchmark_payload() -> dict:
    models = []
    for key, default_name, path in BENCHMARK_SOURCES:
        resolved = resolve(path)
        entry = {"key": key, "name": default_name, "available": resolved.exists(), "path": str(path.name)}
        if resolved.exists():
            with open(resolved, encoding="utf-8") as f:
                data = json.load(f)
            entry["eval"] = data
            entry["name"] = data.get("model", default_name)
        models.append(entry)
    return {"models": models}
