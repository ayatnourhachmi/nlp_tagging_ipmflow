"""Project paths — data, artifacts, and config locations."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
CONFIGS_DIR = PROJECT_ROOT / "configs"

DATASET_PATH = PROCESSED_DATA_DIR / "ipm_dataset.json"
SPLIT_PATH = SPLITS_DIR / "seed42.json"

# Model C — TF-IDF + Logistic Regression
MODEL_C_ARTIFACT = ARTIFACTS_DIR / "ipm_model_c.pkl"
MODEL_C_EVAL = ARTIFACTS_DIR / "ipm_model_c_eval.json"

# Model D — DistilBERT (one checkpoint per dimension)
MODEL_D_CHECKPOINTS = {
    "objective": ARTIFACTS_DIR / "ipm_model_d_objective.pt",
    "domain": ARTIFACTS_DIR / "ipm_model_d_domain.pt",
    "impact": ARTIFACTS_DIR / "ipm_model_d_impact.pt",
    "origin": ARTIFACTS_DIR / "ipm_model_d_origin.pt",
}
MODEL_D_EVAL = ARTIFACTS_DIR / "ipm_model_d_eval.json"

# Model A & B — LLM benchmark results
GROQ_EVAL = ARTIFACTS_DIR / "ipm_groq_eval.json"
OPENAI_EVAL = ARTIFACTS_DIR / "ipm_openai_eval.json"
ALL_EVAL = ARTIFACTS_DIR / "ipm_all_eval.json"


def resolve(path: Path) -> Path:
    """Return path as-is (kept for call-site compatibility)."""
    return path
