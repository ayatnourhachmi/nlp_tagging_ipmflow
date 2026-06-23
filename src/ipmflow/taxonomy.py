"""IPM Flow taxonomy — label definitions shared across all models."""

OBJECTIVE_CLASSES = [
    "cost_reduction",
    "cx_improvement",
    "risk_mitigation",
    "market_opportunity",
]

DOMAIN_CLASSES = [
    "AI",
    "Cloud",
    "Cybersecurity",
    "Data",
    "HR",
    "Finance",
    "Operations",
    "Other",
]

IMPACT_CLASSES = [
    "Revenue",
    "Cost",
    "Risk",
    "CustomerExperience",
]

ORIGIN_CLASSES = [
    "market_driver",
    "operational_problem",
    "client_request",
]

DIMENSIONS = ("objective", "domain", "impact", "origin")

MULTILABEL_THRESHOLD = 0.35
CONFIDENCE_HIGH = 0.72
CONFIDENCE_MEDIUM = 0.45


def confidence_band(probability: float) -> str:
    if probability > CONFIDENCE_HIGH:
        return "high"
    if probability > CONFIDENCE_MEDIUM:
        return "medium"
    return "low"
