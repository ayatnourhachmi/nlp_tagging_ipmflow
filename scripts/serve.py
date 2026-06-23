#!/usr/bin/env python
"""Start the benchmark dashboard and LLM proxy server."""

import _bootstrap  # noqa: F401

from ipmflow.serve.server import run

if __name__ == "__main__":
    run()
