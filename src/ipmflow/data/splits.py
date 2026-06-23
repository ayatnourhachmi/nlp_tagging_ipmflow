"""Canonical stratified train/test split — shared by all models."""

import json

from sklearn.model_selection import train_test_split

from ipmflow.config import load_config
from ipmflow.data.load import load_dataset
from ipmflow.paths import SPLIT_PATH


def create_split(dataset: list[dict] | None = None, save: bool = True) -> dict:
    cfg = load_config()["split"]
    dataset = dataset or load_dataset()

    idx = list(range(len(dataset)))
    stratify = [d["labels"][cfg["stratify_on"]] for d in dataset]
    idx_train, idx_test = train_test_split(
        idx,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=stratify,
    )

    split = {
        "random_state": cfg["random_state"],
        "test_size": cfg["test_size"],
        "stratify_on": cfg["stratify_on"],
        "n_total": len(dataset),
        "n_train": len(idx_train),
        "n_test": len(idx_test),
        "idx_train": sorted(idx_train),
        "idx_test": sorted(idx_test),
    }

    if save:
        SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SPLIT_PATH, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)

    return split


def ensure_split() -> dict:
    if SPLIT_PATH.exists():
        with open(SPLIT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return create_split()
