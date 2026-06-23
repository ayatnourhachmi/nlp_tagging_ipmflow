#!/usr/bin/env python
"""Benchmark LLM and/or trained models on the held-out test set."""

import argparse

import _bootstrap  # noqa: F401

from ipmflow.eval.benchmark import (
    benchmark_all,
    benchmark_llm,
    benchmark_trained,
    save_result,
)
from ipmflow.paths import ALL_EVAL, GROQ_EVAL, OPENAI_EVAL


def main():
    parser = argparse.ArgumentParser(description="Evaluate IPM Flow models")
    parser.add_argument(
        "--models",
        default="llm,trained",
        help="Comma-separated: llm, groq, openai, c, d, trained, all (default: llm,trained)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit to N test pitches (0 = all)")
    args = parser.parse_args()

    selected = {m.strip().lower() for m in args.models.split(",")}

    if "all" in selected:
        result = benchmark_all(args.limit)
        save_result(result, ALL_EVAL)
        return

    if "llm" in selected or "groq" in selected:
        save_result(benchmark_llm("groq", args.limit), GROQ_EVAL)
    if "llm" in selected or "openai" in selected:
        save_result(benchmark_llm("openai", args.limit), OPENAI_EVAL)

    trained = [m for m in ("c", "d") if m in selected or "trained" in selected]
    if trained:
        for result in benchmark_trained(trained, args.limit):
            print(f"\n{result['model']}: avg F1 = {result['avg_f1']}")


if __name__ == "__main__":
    main()
