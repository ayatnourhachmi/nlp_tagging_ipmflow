#!/usr/bin/env python
"""Train a supervised model (C or D)."""

import argparse

import _bootstrap  # noqa: F401

from ipmflow.data.splits import ensure_split
from ipmflow.models import distilbert, sklearn_tfidf

TRAINERS = {
    "c": ("Model C — TF-IDF + Logistic Regression", sklearn_tfidf.train),
    "d": ("Model D — DistilBERT", distilbert.train),
}


def main():
    parser = argparse.ArgumentParser(description="Train IPM Flow classifiers")
    parser.add_argument("model", choices=TRAINERS.keys(), help="Model to train: c or d")
    args = parser.parse_args()

    ensure_split()
    label, trainer = TRAINERS[args.model]
    print(f"Training {label}...")
    trainer()


if __name__ == "__main__":
    main()
