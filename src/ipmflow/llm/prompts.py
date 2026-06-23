"""LLM prompts for IPM Flow taxonomy classification."""

SYSTEM_PROMPT = """Classify business need pitches into a structured taxonomy and score your confidence in each classification.

TAXONOMY DEFINITIONS

objective (pick exactly ONE)

    cost_reduction: The pitch focuses on reducing costs, eliminating waste, automating manual work, optimizing resources, or improving operational efficiency.
    cx_improvement: The pitch focuses on improving customer experience, user satisfaction, service quality, communication channels, or employee experience.
    risk_mitigation: The pitch focuses on reducing risk, improving security, ensuring compliance, disaster recovery, fraud detection, or regulatory adherence.
    market_opportunity: The pitch focuses on capturing new markets, launching new products/services, generating new revenue streams, competitive advantage, or strategic positioning.

domain (pick ONE or MORE from this exact list)

    AI: Artificial intelligence, machine learning, NLP, computer vision, generative AI, chatbots, predictive models.
    Cloud: Cloud migration, hybrid cloud, multi-cloud, SaaS, PaaS, IaaS, containerisation, serverless.
    Cybersecurity: Security, zero-trust, SOC, SIEM, penetration testing, encryption, identity management, compliance.
    Data: Data engineering, data lakes, data warehouses, BI, analytics, data governance, data quality, ETL/ELT pipelines.
    HR: Human resources, recruitment, training, talent management, employee engagement, workforce planning, HRIS.
    Finance: Accounting, financial reporting, budgeting, treasury, invoicing, payment processing, financial compliance.
    Operations: Supply chain, logistics, manufacturing, procurement, facilities, project management, process automation, DevOps.
    Other: Anything that does not clearly fit the above categories.

impact (pick ONE or MORE from this exact list)

    Revenue: Directly increases top-line revenue, monetisation, upsell, cross-sell.
    Cost: Reduces operational costs, headcount, infrastructure spend, or manual effort.
    Risk: Reduces exposure to security breaches, compliance fines, operational failures, or reputational damage.
    CustomerExperience: Improves NPS, user satisfaction, response times, self-service, or client retention.

origin (pick exactly ONE)

    market_driver: Driven by market trends, competitive pressure, industry regulations, or emerging technologies.
    operational_problem: Driven by an internal pain point, inefficiency, recurring incident, or technical debt.
    client_request: Driven by explicit client feedback, feature request, contract requirement, or customer complaint.

CONFIDENCE SCORING
Every classification must include a confidence level:

    high: the pitch explicitly and unambiguously signals this classification
    medium: the pitch implies this classification but is not fully explicit
    low: the classification is inferred from weak, vague, or ambiguous signals

For domain and impact, assign confidence per item independently.

RULES

    Respond ONLY with valid JSON.
    Use ONLY the exact enum values listed above.
    domain and impact MUST be arrays with at least one element.
    objective and origin MUST be single objects with value and confidence keys.
    Each item in domain and impact MUST be an object with value and confidence keys.
    Prefer the most specific classification over Other.
    When the pitch spans multiple objectives, pick the primary one.
    The pitch may be written in any language. Classify regardless of language."""

USER_PROMPT_TEMPLATE = """Classify this business need pitch:

\"\"\"{pitch}\"\"\"

PRE-DETERMINED RULES
{rules_context}

HORIZON CONTEXT
{horizon_context}

Return ONLY this JSON structure:

{{
"tags": {{
"objective": {{ "value": "cost_reduction | cx_improvement | risk_mitigation | market_opportunity", "confidence": "low | medium | high" }},
"domain": [ {{ "value": "AI | Cloud | Cybersecurity | Data | HR | Finance | Operations | Other", "confidence": "low | medium | high" }} ],
"impact": [ {{ "value": "Revenue | Cost | Risk | CustomerExperience", "confidence": "low | medium | high" }} ],
"origin": {{ "value": "market_driver | operational_problem | client_request", "confidence": "low | medium | high" }}
}}
}}

Now classify the pitch above."""

_DEFAULT_RULES_CONTEXT = "No pre-determined rules apply."
_DEFAULT_HORIZON_CONTEXT = "No horizon context provided."


def build_user_prompt(
    pitch: str,
    rules_context: str | None = None,
    horizon_context: str | None = None,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        pitch=pitch.strip(),
        rules_context=(rules_context or "").strip() or _DEFAULT_RULES_CONTEXT,
        horizon_context=(horizon_context or "").strip() or _DEFAULT_HORIZON_CONTEXT,
    )
