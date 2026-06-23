"""Load experiment configuration from YAML."""

from pathlib import Path
from typing import Any

import yaml

from ipmflow.paths import CONFIGS_DIR

DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return _default_config()
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _default_config() -> dict[str, Any]:
    return {
        "split": {"test_size": 0.2, "random_state": 42, "stratify_on": "objective"},
        "model_c": {
            "max_features": 5000,
            "C": 5.0,
            "max_iter": 1000,
            "ngram_range": [1, 1],
            "sublinear_tf": True,
            "stop_words": "english",
            "min_df": 1,
        },
        "model_d": {
            "model_name": "distilbert-base-uncased",
            "max_len": 128,
            "batch_size": 16,
            "epochs": 20,
            "lr": 2e-5,
            "warmup_ratio": 0.1,
            "dropout": 0.2,
            "weight_decay": 0.01,
            "val_ratio": 0.15,
            "early_stopping_patience": 4,
        },
    }
