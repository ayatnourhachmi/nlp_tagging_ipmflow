#!/usr/bin/env python
"""Generate the IPM Flow labeled dataset."""

import json

import _bootstrap  # noqa: F401
from ipmflow.data.generate import TARGET_N, build_dataset, print_stats
from ipmflow.data.splits import create_split
from ipmflow.paths import DATASET_PATH


def main():
    dataset = build_dataset(TARGET_N)
    print_stats(dataset)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {DATASET_PATH}")
    split = create_split(dataset)
    print(f"Split saved: {split['n_train']} train / {split['n_test']} test")


if __name__ == "__main__":
    main()
