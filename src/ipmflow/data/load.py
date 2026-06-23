"""Load dataset and canonical train/test splits."""

import json
from pathlib import Path

from ipmflow.paths import DATASET_PATH, SPLIT_PATH, resolve


def load_dataset(path: Path | None = None) -> list[dict]:
    dataset_path = resolve(path or DATASET_PATH)
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)


def load_split(path: Path | None = None) -> dict:
    split_path = resolve(path or SPLIT_PATH)
    with open(split_path, encoding="utf-8") as f:
        return json.load(f)


def get_texts(dataset: list[dict]) -> list[str]:
    return [d["text"] for d in dataset]


def get_labels(dataset: list[dict]) -> list[dict]:
    return [d["labels"] for d in dataset]
