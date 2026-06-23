"""Unified inference entry point."""

from ipmflow.models import distilbert, sklearn_tfidf


def predict(text: str, model: str = "c") -> dict:
    model = model.lower()
    if model == "c":
        artifact = sklearn_tfidf.load_artifact()
        return sklearn_tfidf.predict(text, artifact)
    if model == "d":
        return distilbert.predict(text)
    raise ValueError(f"Unknown model '{model}'. Choose from: c, d")
