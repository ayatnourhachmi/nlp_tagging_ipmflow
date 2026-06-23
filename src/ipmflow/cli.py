"""Console entry points for pip-installed commands."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def generate():
    from scripts.generate_data import main
    main()


def train():
    from scripts.train import main
    main()


def predict():
    from scripts.predict import main
    main()


def evaluate():
    from scripts.evaluate import main
    main()


def serve():
    from scripts.serve import main
    main()
