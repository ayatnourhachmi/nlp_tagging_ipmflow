#!/usr/bin/env python
"""Run inference on one or more pitches."""

import argparse
import json

import _bootstrap  # noqa: F401

from ipmflow.inference.predict import predict


def main():
    parser = argparse.ArgumentParser(description="Classify IPM Flow pitches")
    parser.add_argument("text", nargs="?", help="Pitch text (or use --file)")
    parser.add_argument("--model", choices=["c", "d"], default="c")
    parser.add_argument("--file", help="Read pitch from file")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read().strip()
    elif args.text:
        text = args.text
    else:
        text = (
            "Automate monthly account reconciliation with an RPA tool to eliminate "
            "manual errors and reduce the close cycle from 5 days to 1 day."
        )

    result = predict(text, model=args.model)
    print(json.dumps({k: v for k, v in result.items() if k != "_meta"}, indent=2, ensure_ascii=False))
    if "_meta" in result:
        print(f"\nLatency: {result['_meta']['latency_ms']}ms  |  model: {result['_meta']['model']}")


if __name__ == "__main__":
    main()
