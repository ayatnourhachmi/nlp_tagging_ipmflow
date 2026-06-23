"""
Step 1 — Generate annotated pitches for the REAL IPM Flow taxonomy.

Sources:
  - REAL_PITCHES: realistic enterprise wording (~43)
  - TEMPLATES: synthetic coverage across taxonomy (~170)
  - EDGE_CASE_PITCHES: ambiguous / multi-signal phrasing (~22)
  - Paraphrase augmentation boosts weak classes (Cybersecurity, HR, Finance, client_request)

Taxonomy:
  objective: cost_reduction | cx_improvement | risk_mitigation | market_opportunity
  domain:    AI | Cloud | Cybersecurity | Data | HR | Finance | Operations | Other
  impact:    Revenue | Cost | Risk | CustomerExperience
  origin:    market_driver | operational_problem | client_request
"""

import random
import re
from collections import Counter

random.seed(42)

TARGET_N = 500

# Each entry: (text_en, objective, domains[], impacts[], origin)
# fmt: off

REAL_PITCHES = [
    # --- Client-driven (client_request) ---
    ("Our top insurance client requested real-time claim status on the portal — legal signed off, delivery expected in Q3.",
     "cx_improvement", ["Operations", "Cloud"], ["CustomerExperience"], "client_request"),
    ("Three enterprise accounts asked for SSO integration with their identity provider before renewing contracts.",
     "cx_improvement", ["Cybersecurity", "Cloud"], ["CustomerExperience", "Risk"], "client_request"),
    ("The retail partner consortium wants a shared analytics view of sell-through data — this came up in every QBR.",
     "cx_improvement", ["Data", "Cloud"], ["CustomerExperience", "Revenue"], "client_request"),
    ("Customer success escalated repeated complaints about slow refund processing — we need a visible SLA workflow.",
     "cx_improvement", ["Operations", "Finance"], ["CustomerExperience"], "client_request"),
    ("A major hospital client asked for HL7/FHIR interoperability to sync patient billing with their EHR.",
     "cx_improvement", ["Operations", "Data"], ["CustomerExperience", "Revenue"], "client_request"),
    ("Procurement flagged that suppliers want electronic PO confirmation — buyers are pushing us to deliver it.",
     "cx_improvement", ["Operations", "Finance"], ["CustomerExperience", "Cost"], "client_request"),
    ("Field sales teams requested mobile access to pricing and inventory during client visits — high churn risk if ignored.",
     "cx_improvement", ["Cloud", "Operations"], ["CustomerExperience", "Revenue"], "client_request"),
    ("Two public-sector clients require accessibility compliance (WCAG 2.1) on all citizen-facing forms by next audit.",
     "cx_improvement", ["Operations", "Cybersecurity"], ["CustomerExperience", "Risk"], "client_request"),

    # --- Finance (weak class) ---
    ("Treasury still reconciles bank feeds manually in Excel — CFO wants straight-through processing by year-end.",
     "cost_reduction", ["Finance", "Data"], ["Cost", "Risk"], "operational_problem"),
    ("Audit found duplicate vendor payments totaling €400K — we need automated three-way matching on invoices.",
     "risk_mitigation", ["Finance", "Operations"], ["Risk", "Cost"], "operational_problem"),
    ("Working capital is strained: collections take 62 days on average — deploy dunning automation and cash forecasting.",
     "cost_reduction", ["Finance", "AI"], ["Cost", "Revenue"], "operational_problem"),
    ("Intercompany billing across 14 entities takes 12 days each close — finance wants a single consolidation hub.",
     "cost_reduction", ["Finance", "Data"], ["Cost"], "operational_problem"),
    ("The FP&A team cannot scenario-plan quickly enough for board meetings — need driver-based planning in the cloud.",
     "market_opportunity", ["Finance", "Cloud"], ["Revenue", "Cost"], "market_driver"),
    ("We missed SOX control deadlines twice because journal entries lack automated approval trails.",
     "risk_mitigation", ["Finance", "Operations"], ["Risk"], "operational_problem"),

    # --- HR (weak class) ---
    ("People analytics is blind: headcount, skills, and attrition live in five different systems — CHRO wants one source of truth.",
     "cx_improvement", ["HR", "Data"], ["CustomerExperience", "Cost"], "operational_problem"),
    ("Managers spend 6+ hours per hire on scheduling interviews — TA lead wants automated screening and scheduling.",
     "cost_reduction", ["HR", "AI"], ["Cost", "CustomerExperience"], "operational_problem"),
    ("Union negotiations require transparent shift-swap and overtime tracking — current spreadsheets are error-prone.",
     "risk_mitigation", ["HR", "Operations"], ["Risk", "CustomerExperience"], "client_request"),
    ("Graduate intake doubled but onboarding materials are outdated — employees rated onboarding 2.1/5 in the last survey.",
     "cx_improvement", ["HR", "Cloud"], ["CustomerExperience"], "client_request"),
    ("Skills gaps in cloud and cybersecurity roles are blocking project delivery — build an internal upskilling academy.",
     "cx_improvement", ["HR", "Cybersecurity"], ["CustomerExperience", "Risk"], "market_driver"),

    # --- Cybersecurity (weak class) ---
    ("Pen test report showed critical RCE on the legacy VPN appliance — replace with zero-trust network access.",
     "risk_mitigation", ["Cybersecurity", "Cloud"], ["Risk"], "operational_problem"),
    ("DORA comes into force next year — we have no unified view of ICT third-party risk across business units.",
     "risk_mitigation", ["Cybersecurity", "Finance"], ["Risk"], "market_driver"),
    ("Security ops is drowning in false-positive alerts — tune SIEM rules and add SOAR playbooks for tier-1 response.",
     "risk_mitigation", ["Cybersecurity", "AI"], ["Risk", "Cost"], "operational_problem"),
    ("Developers push secrets to GitHub weekly — implement secret scanning and vault-based credential rotation.",
     "risk_mitigation", ["Cybersecurity", "Cloud"], ["Risk"], "operational_problem"),
    ("Board mandate: all privileged admin sessions must be recorded and reviewed — PAM tooling not in place today.",
     "risk_mitigation", ["Cybersecurity"], ["Risk"], "market_driver"),
    ("Phishing click rate is 18% in the last simulation — mandatory awareness program with targeted coaching.",
     "risk_mitigation", ["Cybersecurity", "HR"], ["Risk"], "operational_problem"),

    # --- Mixed / operational realism ---
    ("Warehouse pick accuracy dropped to 94% after SKU catalog growth — voice-picking or AR guidance pilot needed.",
     "cx_improvement", ["Operations", "AI"], ["CustomerExperience", "Cost"], "operational_problem"),
    ("Marketing wants a CDP to unify web, CRM, and POS data for campaigns — IT concerned about consent management.",
     "market_opportunity", ["Data", "Cloud"], ["Revenue", "Risk"], "market_driver"),
    ("Manufacturing OEE stalled at 68% — integrate machine telemetry with MES for downtime root-cause analysis.",
     "cost_reduction", ["Operations", "Data"], ["Cost", "Revenue"], "operational_problem"),

    # --- AI / Cloud / Data ---
    ("Legal team spends 3 hours per contract on clause extraction — pilot generative AI review to cut that to 20 minutes.",
     "cost_reduction", ["AI", "Operations"], ["Cost", "Revenue"], "operational_problem"),
    ("Five regional master data registries are out of sync — a central MDM platform is blocking the CRM rollout.",
     "cost_reduction", ["Data", "Cloud"], ["Cost", "Risk"], "operational_problem"),
    ("Cloud security posture review found 140 misconfigured S3 buckets — CSPM tooling needed before next board audit.",
     "risk_mitigation", ["Cloud", "Cybersecurity"], ["Risk"], "operational_problem"),
    ("Data science team rebuilds feature pipelines from scratch each project — a shared feature store would cut model time-to-prod by half.",
     "cost_reduction", ["AI", "Data"], ["Cost", "Revenue"], "operational_problem"),
    ("FinOps analysis showed 42% of cloud spend is waste — tagging enforcement, rightsizing, and reservation planning required.",
     "cost_reduction", ["Cloud", "Finance"], ["Cost"], "operational_problem"),
    ("MLOps is missing: models trained in notebooks are deployed manually with no monitoring — shadow model risk is high.",
     "risk_mitigation", ["AI", "Cloud"], ["Risk", "Cost"], "operational_problem"),

    # --- HR ---
    ("Remote-work survey shows 34% of employees feel isolated — need a digital well-being and connection platform.",
     "cx_improvement", ["HR", "Cloud"], ["CustomerExperience"], "client_request"),
    ("HR analytics shows voluntary attrition in engineering is 28% — need predictive models to flag flight risks early.",
     "cost_reduction", ["HR", "Data"], ["Cost", "Risk"], "operational_problem"),
    ("New CSRD directive requires gender pay-gap reporting by next April — payroll data is too fragmented to produce it today.",
     "risk_mitigation", ["HR", "Finance"], ["Risk"], "market_driver"),

    # --- Finance ---
    ("Credit risk models are five years old and built on pre-pandemic data — re-training needed before the Basel IV deadline.",
     "risk_mitigation", ["Finance", "AI"], ["Risk"], "market_driver"),
    ("Transfer pricing documentation for 22 jurisdictions is assembled manually each year — automate before the OECD Pillar Two audit.",
     "risk_mitigation", ["Finance", "Operations"], ["Risk", "Cost"], "market_driver"),

    # --- Operations / Market ---
    ("Clients can't see where their shipment is until it's delivered — real-time logistics visibility portal is a contract differentiator.",
     "cx_improvement", ["Operations", "Data"], ["CustomerExperience", "Revenue"], "client_request"),
    ("Digital twin of the main plant would let engineers test production scenarios without halting the line.",
     "market_opportunity", ["Operations", "AI"], ["Revenue", "Cost"], "market_driver"),
    ("Embedded analytics in our SaaS product is the top feature request — adding it would unlock the enterprise tier upsell.",
     "market_opportunity", ["AI", "Data"], ["Revenue", "CustomerExperience"], "client_request"),
    ("AI-driven threat hunting is reactive today — security team wants proactive hunt cycles using behavioral baselines.",
     "risk_mitigation", ["Cybersecurity", "AI"], ["Risk"], "operational_problem"),
]

EDGE_CASE_PITCHES = [
    # Deliberately mixed signals — gold label reflects primary intent
    ("Consolidate CRM and billing platforms to cut SaaS spend while giving clients a single self-service account view.",
     "cost_reduction", ["Cloud", "Finance"], ["Cost", "CustomerExperience"], "operational_problem"),
    ("Use generative AI to draft support replies faster — goal is both cost takeout and higher CSAT scores.",
     "cost_reduction", ["AI", "Operations"], ["Cost", "CustomerExperience"], "operational_problem"),
    ("Expand into Southeast Asia before competitors — requires localized payments, tax, and bilingual support.",
     "market_opportunity", ["Finance", "Operations"], ["Revenue", "CustomerExperience"], "market_driver"),
    ("Client contract mandates 99.95% uptime and sub-4h incident comms — current ops model cannot guarantee it.",
     "risk_mitigation", ["Operations", "Cloud"], ["Risk", "CustomerExperience"], "client_request"),
    ("Replace legacy ERP modules gradually vs big-bang — finance and supply chain disagree on sequencing.",
     "cost_reduction", ["Finance", "Operations"], ["Cost", "Risk"], "operational_problem"),
    ("Pilot copilot for engineers: productivity gain unclear but talent retention argument is strong.",
     "cx_improvement", ["AI", "HR"], ["CustomerExperience", "Cost"], "market_driver"),
    ("Reduce cloud spend by 20% without slowing product releases — FinOps tooling and architecture review.",
     "cost_reduction", ["Cloud", "Finance"], ["Cost"], "operational_problem"),
    ("Insurance regulator asked for explainable AI on underwriting models — compliance deadline in 9 months.",
     "risk_mitigation", ["AI", "Finance"], ["Risk"], "market_driver"),
    ("Standardize HR and IT service desk on one platform — employee experience vs integration cost tradeoff.",
     "cx_improvement", ["HR", "Operations"], ["CustomerExperience", "Cost"], "client_request"),
    ("Acquire a fintech API startup vs build in-house payment rails — strategy team wants decision support.",
     "market_opportunity", ["Finance", "Other"], ["Revenue", "Risk"], "market_driver"),
    ("Shift-left security in CI/CD: developers resist friction but breach risk on public-facing apps is rising.",
     "risk_mitigation", ["Cybersecurity", "Cloud"], ["Risk", "Cost"], "market_driver"),
    ("NPS is up but support costs doubled — automate tier-1 without hurting the premium brand experience.",
     "cost_reduction", ["AI", "Operations"], ["Cost", "CustomerExperience"], "operational_problem"),
    ("New CFO wants simultaneous cost cuts and digital transformation — current planning cycle can't handle both priorities.",
     "cost_reduction", ["Finance", "Operations"], ["Cost", "Revenue"], "operational_problem"),
    ("Clients demand better data privacy controls, but sales argues the added friction will hurt conversion rates.",
     "risk_mitigation", ["Cybersecurity", "Data"], ["Risk", "CustomerExperience"], "client_request"),
    ("We want to launch in the US market, but compliance, tax, and support localization extend timelines by 18 months.",
     "market_opportunity", ["Finance", "Operations"], ["Revenue", "Risk"], "market_driver"),
    ("Replace an aging field sales force with inside sales augmented by AI — significant culture-change risk involved.",
     "cost_reduction", ["AI", "HR"], ["Cost", "Revenue"], "operational_problem"),
    ("ESG investors want quarterly scope-3 emissions data, but our suppliers resist sharing operational figures.",
     "risk_mitigation", ["Operations", "Data"], ["Risk"], "market_driver"),
    ("Offer a freemium tier to grow the user base faster, even though it dilutes revenue per seat in the short term.",
     "market_opportunity", ["Cloud", "Operations"], ["Revenue", "Cost"], "market_driver"),
    ("Security wants to block all USB drives; field technicians say they need them to service equipment off-grid.",
     "risk_mitigation", ["Cybersecurity", "Operations"], ["Risk", "CustomerExperience"], "operational_problem"),
    ("Build vs buy for the ML platform: cloud vendor lock-in vs a 24-month in-house build — CFO can't decide.",
     "cost_reduction", ["AI", "Cloud"], ["Cost", "Risk"], "market_driver"),
    ("Migrate all clients to the new platform version, but 20% are on contractually guaranteed legacy support.",
     "cx_improvement", ["Cloud", "Operations"], ["CustomerExperience", "Risk"], "client_request"),
    ("Launch AI-driven dynamic pricing — revenue management teams love it, but key accounts are threatening churn.",
     "market_opportunity", ["AI", "Finance"], ["Revenue", "CustomerExperience"], "market_driver"),
]

TEMPLATES = [
    # cost_reduction × Operations × operational_problem
    ("Automate monthly account reconciliation with an RPA tool to eliminate manual errors and reduce the close cycle from 5 days to 1 day.",
     "cost_reduction", ["Operations", "Finance"], ["Cost", "Risk"], "operational_problem"),
    ("Implement an invoice processing automation tool to reduce manual handling by 70%.",
     "cost_reduction", ["Finance", "Operations"], ["Cost"], "operational_problem"),
    ("Deploy RPA on supplier purchase order processing to reduce delays and data entry errors.",
     "cost_reduction", ["Operations"], ["Cost", "Risk"], "operational_problem"),
    ("Optimize inventory management to reduce overstock and stockouts, saving $2M per year.",
     "cost_reduction", ["Operations", "Data"], ["Cost"], "operational_problem"),
    ("Migrate on-premise servers to the cloud to cut infrastructure costs by 40%.",
     "cost_reduction", ["Cloud"], ["Cost"], "market_driver"),
    ("Consolidate redundant SaaS tools to reduce licensing costs by 30% and simplify the user experience.",
     "cost_reduction", ["Cloud", "Operations"], ["Cost"], "operational_problem"),
    ("Automate monthly financial report generation via a data pipeline to eliminate 20 hours of manual work.",
     "cost_reduction", ["Finance", "Data"], ["Cost"], "operational_problem"),
    ("Reduce datacenter energy consumption through cloud optimization and AI-driven scheduling algorithms.",
     "cost_reduction", ["Cloud", "AI"], ["Cost"], "market_driver"),
    ("Deploy an HR chatbot to reduce tier-1 HR ticket volume and free up staff capacity.",
     "cost_reduction", ["HR", "AI"], ["Cost"], "operational_problem"),
    ("Implement an automated document management solution to reduce administrative workload.",
     "cost_reduction", ["Operations", "AI"], ["Cost"], "operational_problem"),

    # cx_improvement × client_request
    ("Deploy a conversational AI assistant on the customer portal to reduce response time from 48 hours to 2 hours.",
     "cx_improvement", ["AI"], ["CustomerExperience"], "client_request"),
    ("Personalize product recommendations using machine learning to increase customer satisfaction.",
     "cx_improvement", ["AI", "Data"], ["CustomerExperience", "Revenue"], "client_request"),
    ("Improve the employee experience with a unified, mobile-first HR self-service portal.",
     "cx_improvement", ["HR", "Cloud"], ["CustomerExperience"], "client_request"),
    ("Implement real-time NPS with semantic analysis of customer verbatims using NLP.",
     "cx_improvement", ["AI", "Data"], ["CustomerExperience"], "client_request"),
    ("Redesign the online subscription journey to reduce abandonment rate from 45% to below 20%.",
     "cx_improvement", ["Operations"], ["CustomerExperience", "Revenue"], "client_request"),
    ("Launch a real-time order tracking portal to improve transparency and reduce support calls.",
     "cx_improvement", ["Operations", "Cloud"], ["CustomerExperience", "Cost"], "client_request"),
    ("Deploy an omnichannel customer feedback tool to centralize insights and accelerate corrective actions.",
     "cx_improvement", ["Data", "AI"], ["CustomerExperience"], "client_request"),
    ("Improve customer support accessibility via a multilingual chatbot available 24/7.",
     "cx_improvement", ["AI"], ["CustomerExperience"], "client_request"),
    ("Modernize the mobile customer app to deliver a seamless experience and reduce bugs by 60%.",
     "cx_improvement", ["Cloud", "Operations"], ["CustomerExperience"], "operational_problem"),
    ("Implement a digital loyalty program to improve customer retention by 15%.",
     "cx_improvement", ["Data", "AI"], ["CustomerExperience", "Revenue"], "client_request"),

    # risk_mitigation × Cybersecurity
    ("Deploy a zero-trust architecture and managed SOC to secure remote access post-COVID.",
     "risk_mitigation", ["Cybersecurity"], ["Risk"], "market_driver"),
    ("Bring the information system into GDPR compliance before the Q2 regulatory audit.",
     "risk_mitigation", ["Cybersecurity", "Data"], ["Risk"], "market_driver"),
    ("Implement a SIEM to detect intrusions in real time and reduce MTTD to under 1 hour.",
     "risk_mitigation", ["Cybersecurity"], ["Risk"], "market_driver"),
    ("Establish a cloud-based disaster recovery plan to guarantee an RTO under 4 hours.",
     "risk_mitigation", ["Cloud", "Cybersecurity"], ["Risk"], "operational_problem"),
    ("Conduct an annual security audit and penetration test to identify critical vulnerabilities.",
     "risk_mitigation", ["Cybersecurity"], ["Risk"], "market_driver"),
    ("Implement centralized identity and access management (IAM) to reduce intrusion risk.",
     "risk_mitigation", ["Cybersecurity"], ["Risk"], "operational_problem"),
    ("Encrypt sensitive customer data in transit and at rest to comply with ISO 27001.",
     "risk_mitigation", ["Cybersecurity", "Data"], ["Risk"], "market_driver"),
    ("Automate fraud detection on financial transactions using a real-time ML model.",
     "risk_mitigation", ["AI", "Finance"], ["Risk", "Cost"], "operational_problem"),
    ("Deploy a regulatory compliance management tool to avoid penalties exceeding $5M.",
     "risk_mitigation", ["Finance", "Operations"], ["Risk", "Cost"], "market_driver"),
    ("Implement continuous third-party (vendor) monitoring to reduce supply chain risk.",
     "risk_mitigation", ["Operations", "Data"], ["Risk"], "market_driver"),

    # market_opportunity
    ("Launch an AI-based credit scoring offering to address the underserved SMB market.",
     "market_opportunity", ["AI", "Finance"], ["Revenue"], "market_driver"),
    ("Develop a data-as-a-service platform to monetize proprietary data with third parties.",
     "market_opportunity", ["Data", "Cloud"], ["Revenue"], "market_driver"),
    ("Launch an e-commerce vertical to capture 10% of the regional market within 18 months.",
     "market_opportunity", ["Operations", "Cloud"], ["Revenue"], "market_driver"),
    ("Integrate a generative AI content module into our SaaS product to differentiate from competitors.",
     "market_opportunity", ["AI", "Cloud"], ["Revenue"], "market_driver"),
    ("Deploy an industrial predictive maintenance solution to enter the Industry 4.0 market.",
     "market_opportunity", ["AI", "Operations"], ["Revenue"], "market_driver"),
    ("Launch a technology partnership program to accelerate international expansion.",
     "market_opportunity", ["Other"], ["Revenue"], "market_driver"),
    ("Create an internal innovation lab to rapidly prototype new digital offerings.",
     "market_opportunity", ["AI", "Cloud"], ["Revenue"], "market_driver"),
    ("Develop an open API to let partners integrate our services and generate new revenue.",
     "market_opportunity", ["Cloud", "Data"], ["Revenue"], "market_driver"),
    ("Launch a telemedicine solution to address the fast-growing remote healthcare market.",
     "market_opportunity", ["Cloud", "AI"], ["Revenue"], "market_driver"),
    ("Implement a B2B loyalty program with usage-based personalized offers.",
     "market_opportunity", ["Data", "AI"], ["Revenue", "CustomerExperience"], "market_driver"),

    # Extra synthetic — Cybersecurity boost
    ("Roll out endpoint detection and response (EDR) across all laptops and servers within two quarters.",
     "risk_mitigation", ["Cybersecurity"], ["Risk"], "market_driver"),
    ("Segment the OT network from corporate IT to reduce exposure of production systems to ransomware.",
     "risk_mitigation", ["Cybersecurity", "Operations"], ["Risk"], "operational_problem"),
    ("Deploy data loss prevention (DLP) on email and cloud storage to stop exfiltration of PII.",
     "risk_mitigation", ["Cybersecurity", "Data"], ["Risk"], "market_driver"),
    ("Achieve PCI-DSS certification for the payments platform before the holiday sales peak.",
     "risk_mitigation", ["Cybersecurity", "Finance"], ["Risk", "Revenue"], "market_driver"),
    ("Implement bug bounty and responsible disclosure program to harden public APIs.",
     "risk_mitigation", ["Cybersecurity", "Cloud"], ["Risk"], "market_driver"),
    ("Replace shared admin passwords on critical databases with vault-managed rotating credentials.",
     "risk_mitigation", ["Cybersecurity", "Data"], ["Risk"], "operational_problem"),

    # Extra synthetic — Finance boost
    ("Introduce robotic invoice matching linked to ERP to eliminate manual AP clerk rework.",
     "cost_reduction", ["Finance", "Operations"], ["Cost"], "operational_problem"),
    ("Build a liquidity dashboard pulling from all bank APIs for daily treasury decisions.",
     "risk_mitigation", ["Finance", "Data"], ["Risk", "Cost"], "operational_problem"),
    ("Automate VAT and e-invoicing compliance for EU subsidiaries under the new mandate.",
     "risk_mitigation", ["Finance", "Operations"], ["Risk"], "market_driver"),
    ("Offer embedded lending to marketplace sellers — new fee income stream from transaction data.",
     "market_opportunity", ["Finance", "Data"], ["Revenue"], "market_driver"),
    ("Reduce FX hedging errors by integrating trading platform with the general ledger in real time.",
     "risk_mitigation", ["Finance"], ["Risk", "Cost"], "operational_problem"),
    ("Digitize expense reports with OCR and policy checks to cut reimbursement cycle from 14 to 3 days.",
     "cost_reduction", ["Finance", "AI"], ["Cost", "CustomerExperience"], "operational_problem"),

    # Extra synthetic — RH boost
    ("Launch an internal talent marketplace so employees can apply to short-term project gigs.",
     "cx_improvement", ["HR", "Cloud"], ["CustomerExperience"], "client_request"),
    ("Deploy pulse surveys with sentiment analysis to detect burnout in high-turnover teams.",
     "cx_improvement", ["HR", "AI"], ["CustomerExperience", "Risk"], "client_request"),
    ("Automate workday scheduling for shift workers with fairness rules and legal break compliance.",
     "cost_reduction", ["HR", "Operations"], ["Cost", "Risk"], "operational_problem"),
    ("Create a leadership development track with personalized learning paths for high potentials.",
     "cx_improvement", ["HR", "AI"], ["CustomerExperience"], "market_driver"),
    ("Integrate payroll with time tracking to eliminate duplicate data entry for hourly staff.",
     "cost_reduction", ["HR", "Finance"], ["Cost"], "operational_problem"),
    ("Offer employees a benefits portal with AI-guided choices during open enrollment.",
     "cx_improvement", ["HR", "AI"], ["CustomerExperience"], "client_request"),

    # Extra synthetic — client_request boost
    ("Municipality RFP requires citizen appointment booking online — current phone-only process fails the spec.",
     "cx_improvement", ["Operations", "Cloud"], ["CustomerExperience"], "client_request"),
    ("Key account threatened churn unless we add bulk CSV export to their procurement integration.",
     "cx_improvement", ["Operations", "Data"], ["CustomerExperience", "Revenue"], "client_request"),
    ("Wholesale buyers asked for configurable catalog views per region — blocking renewal of master agreement.",
     "cx_improvement", ["Cloud", "Operations"], ["CustomerExperience", "Revenue"], "client_request"),
    ("Patient advocacy group requested plain-language billing summaries — board approved as reputational priority.",
     "cx_improvement", ["Finance", "Operations"], ["CustomerExperience"], "client_request"),
    ("Developers on our platform want webhook retries and signing — top request on the public roadmap forum.",
     "cx_improvement", ["Cloud", "Cybersecurity"], ["CustomerExperience", "Revenue"], "client_request"),
    ("Hotel chain client needs multi-property reporting in one login — deal size €2M annually.",
     "cx_improvement", ["Data", "Cloud"], ["CustomerExperience", "Revenue"], "client_request"),

    # Industries & variety
    ("Build a unified data warehouse to eliminate data silos and reduce duplicate reporting.",
     "cost_reduction", ["Data"], ["Cost"], "operational_problem"),
    ("Automate operational data ETL to reduce reporting lag from D+5 to D+1.",
     "cost_reduction", ["Data", "Operations"], ["Cost"], "operational_problem"),
    ("Deploy a data quality tool to reduce data errors costing $500K per year in manual rework.",
     "cost_reduction", ["Data"], ["Cost", "Risk"], "operational_problem"),
    ("Digitize the new hire onboarding journey to reduce time-to-productivity from 3 months to 6 weeks.",
     "cx_improvement", ["HR", "Cloud"], ["CustomerExperience", "Cost"], "client_request"),
    ("Implement a continuous 360-degree feedback tool to improve employee engagement.",
     "cx_improvement", ["HR", "Data"], ["CustomerExperience"], "client_request"),
    ("Deploy a personalized e-learning platform for sales teams.",
     "cx_improvement", ["HR", "AI"], ["CustomerExperience"], "client_request"),
    ("Automate KYC/AML compliance checks to reduce regulatory risk and onboarding delays.",
     "risk_mitigation", ["Finance", "AI"], ["Risk", "CustomerExperience"], "market_driver"),
    ("Implement an operational risk management solution compliant with Basel IV.",
     "risk_mitigation", ["Finance", "Data"], ["Risk"], "market_driver"),
    ("Deploy a track-and-trace solution to provide full supply chain visibility to customers.",
     "market_opportunity", ["Operations", "Data"], ["Revenue", "CustomerExperience"], "market_driver"),
    ("Launch a next-day express delivery service powered by AI-driven logistics optimization.",
     "market_opportunity", ["Operations", "AI"], ["Revenue", "CustomerExperience"], "market_driver"),
    ("Implement an automated competitive intelligence tool to track market changes in real time.",
     "market_opportunity", ["AI", "Data"], ["Revenue"], "market_driver"),
    ("Deploy an AI-based talent management system to reduce turnover by 25%.",
     "cx_improvement", ["HR", "AI"], ["CustomerExperience", "Cost"], "operational_problem"),
    ("Implement a unified project management platform to improve cross-team collaboration.",
     "cost_reduction", ["Operations", "Cloud"], ["Cost", "CustomerExperience"], "operational_problem"),
    ("Automate sales proposal generation via generative AI to shorten the sales cycle.",
     "cost_reduction", ["AI", "Operations"], ["Cost", "Revenue"], "operational_problem"),
    ("Build a real-time executive dashboard to improve strategic decision-making.",
     "market_opportunity", ["Data", "AI"], ["Revenue"], "operational_problem"),
    ("Deploy a supplier risk management solution to anticipate supply disruptions.",
     "risk_mitigation", ["Operations", "Data"], ["Risk", "Cost"], "market_driver"),
    ("Launch an open innovation program with startups to accelerate digital transformation.",
     "market_opportunity", ["Other"], ["Revenue"], "market_driver"),
    ("Implement an e-signature solution to digitize 100% of customer contracts.",
     "cost_reduction", ["Operations", "Cybersecurity"], ["Cost", "CustomerExperience"], "client_request"),
    ("Deploy dynamic pricing based on demand and competitive data.",
     "market_opportunity", ["AI", "Data"], ["Revenue"], "market_driver"),
    ("Improve the customer claims process to reduce handling time from 10 to 3 days.",
     "cx_improvement", ["Operations", "AI"], ["CustomerExperience", "Cost"], "client_request"),
    ("Implement an accounting anomaly detection tool to prevent internal fraud.",
     "risk_mitigation", ["Finance", "AI"], ["Risk"], "operational_problem"),
    ("Deploy a hybrid cloud infrastructure to improve resilience and reduce datacenter dependency.",
     "risk_mitigation", ["Cloud", "Operations"], ["Risk", "Cost"], "operational_problem"),
    ("Launch a cybersecurity consulting offering for SMBs, an underserved market segment.",
     "market_opportunity", ["Cybersecurity", "Other"], ["Revenue"], "market_driver"),
    ("Automate workforce planning via a predictive model to anticipate hiring needs.",
     "cost_reduction", ["HR", "AI"], ["Cost"], "operational_problem"),
    ("Implement a carbon emissions reduction program with automated ESG reporting.",
     "risk_mitigation", ["Operations", "Data"], ["Risk"], "market_driver"),
    ("Develop a cross-sell recommendation engine to increase average order value by 20%.",
     "market_opportunity", ["AI", "Data"], ["Revenue", "CustomerExperience"], "client_request"),
    ("Deploy a vendor contract management tool to reduce legal risk and optimize costs.",
     "cost_reduction", ["Operations", "Finance"], ["Cost", "Risk"], "operational_problem"),
    ("Launch an instant payment solution for SMBs to capture a new market segment.",
     "market_opportunity", ["Finance", "Cloud"], ["Revenue"], "market_driver"),
    ("Modernize the payroll system to ensure legal compliance and reduce processing errors.",
     "risk_mitigation", ["HR", "Finance"], ["Risk", "Cost"], "operational_problem"),
    ("Deploy an AI assistant for sales teams to accelerate lead qualification.",
     "market_opportunity", ["AI", "Operations"], ["Revenue", "Cost"], "market_driver"),
    ("Build a data lake to centralize sales, marketing, and operations data.",
     "cost_reduction", ["Data", "Cloud"], ["Cost"], "operational_problem"),
    ("Improve IT incident management with an ITSM tool and AI-powered knowledge base.",
     "cx_improvement", ["Operations", "AI"], ["CustomerExperience", "Cost"], "operational_problem"),
    ("Deploy a cyber-resilience program including regular attack simulation exercises.",
     "risk_mitigation", ["Cybersecurity"], ["Risk"], "market_driver"),
    ("Implement a customer collaboration platform (extranet) to improve co-creation.",
     "cx_improvement", ["Cloud", "Operations"], ["CustomerExperience"], "client_request"),
    ("Automate the financial close process to reduce cycle time from 10 days to 3 business days.",
     "cost_reduction", ["Finance", "Operations"], ["Cost"], "operational_problem"),
    ("Launch an AI credit insurance vertical to address high-potential emerging markets.",
     "market_opportunity", ["AI", "Finance"], ["Revenue"], "market_driver"),
    ("Deploy a quality management system (QMS) to reduce product non-conformities by 30%.",
     "risk_mitigation", ["Operations", "Data"], ["Risk", "Cost"], "operational_problem"),
    ("Implement mandatory cybersecurity training for all employees.",
     "risk_mitigation", ["Cybersecurity", "HR"], ["Risk"], "market_driver"),
    ("Develop a SaaS cash management product for mid-market companies, a poorly digitized segment.",
     "market_opportunity", ["Finance", "Cloud"], ["Revenue"], "market_driver"),
    ("Automate ESG data collection and consolidation for regulatory reporting.",
     "risk_mitigation", ["Data", "Operations"], ["Risk", "Cost"], "market_driver"),

    # ── cost_reduction batch 2 ──────────────────────────────────────────────
    ("Automate procurement approval workflows to cut purchase-order cycle time from 10 days to 2.",
     "cost_reduction", ["Operations", "Finance"], ["Cost"], "operational_problem"),
    ("Consolidate three regional data centers into a single cloud provider to reduce OpEx by 35%.",
     "cost_reduction", ["Cloud", "Operations"], ["Cost"], "operational_problem"),
    ("Implement dynamic discounting on supplier invoices to capture early-payment savings worth €1.2M.",
     "cost_reduction", ["Finance", "Operations"], ["Cost"], "operational_problem"),
    ("Replace manual data entry across eight legacy forms with intelligent document processing and validation.",
     "cost_reduction", ["Operations", "AI"], ["Cost", "Risk"], "operational_problem"),
    ("Reduce cold-storage energy use by 22% with IoT sensors and ML-driven cooling schedules.",
     "cost_reduction", ["Operations", "AI"], ["Cost"], "market_driver"),
    ("Automate IT asset lifecycle management to eliminate shadow IT and cut license waste by 30%.",
     "cost_reduction", ["Cloud", "Operations"], ["Cost"], "operational_problem"),
    ("Migrate legacy mainframe batch workloads to cloud-native services to cut overnight processing costs in half.",
     "cost_reduction", ["Cloud", "Operations"], ["Cost"], "operational_problem"),
    ("Deploy AI-powered demand sensing to reduce safety stock by 25% without increasing stockout frequency.",
     "cost_reduction", ["AI", "Operations"], ["Cost"], "operational_problem"),
    ("Implement self-service BI dashboards for department heads to eliminate 90% of ad-hoc data requests.",
     "cost_reduction", ["Data", "Operations"], ["Cost"], "operational_problem"),
    ("Automate IFRS 17 compliance reporting to cut actuarial team effort by 60%.",
     "cost_reduction", ["Finance", "Data"], ["Cost", "Risk"], "market_driver"),
    ("Replace paper-based field inspection reports with a mobile digital capture and sync tool.",
     "cost_reduction", ["Operations", "Cloud"], ["Cost"], "operational_problem"),
    ("Standardize legal contract templates and a shared clause library to cut external legal spend by 40%.",
     "cost_reduction", ["Operations", "Finance"], ["Cost", "Risk"], "operational_problem"),
    ("Shift from reactive to planned maintenance using predictive analytics on equipment sensor data.",
     "cost_reduction", ["Operations", "AI"], ["Cost", "Risk"], "operational_problem"),
    ("Automate end-of-life product returns processing to cut reverse-logistics handling cost by 35%.",
     "cost_reduction", ["Operations", "Finance"], ["Cost"], "operational_problem"),
    ("Deploy a transport management system to optimize carrier selection and reduce freight spend by 18%.",
     "cost_reduction", ["Operations", "Data"], ["Cost"], "operational_problem"),

    # ── cx_improvement batch 2 ─────────────────────────────────────────────
    ("Embed a real-time delivery ETA widget in client portals to reduce inbound tracking calls by 50%.",
     "cx_improvement", ["Operations", "Cloud"], ["CustomerExperience", "Cost"], "client_request"),
    ("Build a guided product selection wizard to reduce customer decision friction and cart abandonment.",
     "cx_improvement", ["Operations", "AI"], ["CustomerExperience", "Revenue"], "client_request"),
    ("Deploy agent-assist AI in the contact center to surface relevant knowledge articles in real time.",
     "cx_improvement", ["AI", "Operations"], ["CustomerExperience", "Cost"], "operational_problem"),
    ("Create a client health-score dashboard for account managers to identify and prioritize at-risk accounts.",
     "cx_improvement", ["Data", "AI"], ["CustomerExperience", "Revenue"], "operational_problem"),
    ("Launch proactive maintenance notifications via SMS and app to reduce service-call escalations by 30%.",
     "cx_improvement", ["Operations", "Cloud"], ["CustomerExperience", "Cost"], "client_request"),
    ("Deploy a configure-price-quote tool to cut B2B sales proposal turnaround from 5 days to same-day.",
     "cx_improvement", ["Operations", "AI"], ["CustomerExperience", "Revenue"], "client_request"),
    ("Modernize the IVR system with conversational AI to reduce call misrouting by 40%.",
     "cx_improvement", ["AI", "Operations"], ["CustomerExperience", "Cost"], "operational_problem"),
    ("Launch a visual product configurator so B2B buyers can self-design and price complex orders.",
     "cx_improvement", ["Cloud", "Operations"], ["CustomerExperience", "Revenue"], "client_request"),
    ("Implement in-app onboarding tutorials to cut time-to-first-value for new SaaS users by half.",
     "cx_improvement", ["Cloud", "AI"], ["CustomerExperience"], "client_request"),
    ("Create a white-label partner portal with branded dashboards for the reseller channel.",
     "cx_improvement", ["Cloud", "Data"], ["CustomerExperience", "Revenue"], "client_request"),
    ("Deploy sentiment analysis on support tickets to flag dissatisfied customers for proactive outreach.",
     "cx_improvement", ["AI", "Data"], ["CustomerExperience"], "operational_problem"),
    ("Launch a customer community platform to reduce support load and increase product adoption scores.",
     "cx_improvement", ["Cloud", "Operations"], ["CustomerExperience", "Cost"], "client_request"),
    ("Build a real-time in-store inventory visibility tool for retail associates to answer stock queries instantly.",
     "cx_improvement", ["Operations", "Data"], ["CustomerExperience"], "client_request"),
    ("Implement a digital concierge app for hotel guests to replace paper-based room-service request processes.",
     "cx_improvement", ["Cloud", "Operations"], ["CustomerExperience"], "client_request"),
    ("Build a citizen-facing complaint-tracking portal to improve transparency and trust in government services.",
     "cx_improvement", ["Cloud", "Operations"], ["CustomerExperience"], "client_request"),

    # ── risk_mitigation batch 2 ────────────────────────────────────────────
    ("Implement a supply chain disruption early-warning system with multi-tier supplier risk mapping.",
     "risk_mitigation", ["Operations", "Data"], ["Risk"], "market_driver"),
    ("Automate SOC 2 evidence collection to reduce audit preparation time from six weeks to five days.",
     "risk_mitigation", ["Cybersecurity", "Cloud"], ["Risk", "Cost"], "market_driver"),
    ("Deploy an API gateway with WAF and rate limiting to protect all public-facing services.",
     "risk_mitigation", ["Cybersecurity", "Cloud"], ["Risk"], "operational_problem"),
    ("Establish a third-party risk-scoring platform integrating external threat intelligence feeds.",
     "risk_mitigation", ["Cybersecurity", "Operations"], ["Risk"], "market_driver"),
    ("Implement automated backup verification to confirm recovery-point objectives are actually met.",
     "risk_mitigation", ["Cloud", "Cybersecurity"], ["Risk"], "operational_problem"),
    ("Deploy an insider threat detection system using user and entity behavior analytics (UEBA).",
     "risk_mitigation", ["Cybersecurity", "Data"], ["Risk"], "market_driver"),
    ("Standardize incident response playbooks and automate containment steps on critical security alerts.",
     "risk_mitigation", ["Cybersecurity", "Operations"], ["Risk"], "operational_problem"),
    ("Implement a model risk management framework covering all production ML models used in business decisions.",
     "risk_mitigation", ["AI", "Finance"], ["Risk"], "market_driver"),
    ("Automate personal data discovery and classification across structured and unstructured data stores for GDPR.",
     "risk_mitigation", ["Data", "Cybersecurity"], ["Risk"], "market_driver"),
    ("Automate open-source license compliance tracking for all software dependencies in production.",
     "risk_mitigation", ["Cybersecurity", "Operations"], ["Risk", "Cost"], "market_driver"),
    ("Deploy real-time transaction monitoring for AML compliance across all payment channels.",
     "risk_mitigation", ["Finance", "AI"], ["Risk"], "market_driver"),
    ("Build a business continuity testing platform to run automated scenario simulations each quarter.",
     "risk_mitigation", ["Operations", "Cloud"], ["Risk"], "operational_problem"),
    ("Implement a document retention and legal-hold management tool to reduce e-discovery costs.",
     "risk_mitigation", ["Operations", "Finance"], ["Risk", "Cost"], "market_driver"),
    ("Deploy a privacy-enhancing computation layer for cross-entity analytics without sharing raw data.",
     "risk_mitigation", ["Data", "Cybersecurity"], ["Risk"], "market_driver"),
    ("Automate change-management controls in the ERP to enforce segregation of duties and audit trails.",
     "risk_mitigation", ["Finance", "Operations"], ["Risk"], "operational_problem"),

    # ── market_opportunity batch 2 ─────────────────────────────────────────
    ("Launch an embedded analytics module within the platform to upsell to the enterprise tier.",
     "market_opportunity", ["AI", "Data"], ["Revenue"], "market_driver"),
    ("Develop a carbon accounting and offset marketplace for enterprise sustainability buyers.",
     "market_opportunity", ["Data", "Cloud"], ["Revenue"], "market_driver"),
    ("Build a no-code workflow automation product for SMBs unable to afford custom software development.",
     "market_opportunity", ["Cloud", "Operations"], ["Revenue"], "market_driver"),
    ("Launch a vertical SaaS offering tailored for healthcare compliance and accreditation management.",
     "market_opportunity", ["Cloud", "Finance"], ["Revenue"], "market_driver"),
    ("Create a developer ecosystem with SDKs and an API marketplace to attract third-party integrations.",
     "market_opportunity", ["Cloud", "Data"], ["Revenue"], "market_driver"),
    ("Expand into the MENA region by localizing the platform for Arabic, regional payment rails, and tax rules.",
     "market_opportunity", ["Finance", "Operations"], ["Revenue"], "market_driver"),
    ("Launch a digital-first insurance product for gig-economy workers — currently an underserved segment.",
     "market_opportunity", ["Finance", "Cloud"], ["Revenue"], "market_driver"),
    ("Build a managed security operations center offering for mid-market companies lacking in-house SecOps.",
     "market_opportunity", ["Cybersecurity", "Cloud"], ["Revenue"], "market_driver"),
    ("Develop an AI-powered legal contract review tool for in-house legal teams in regulated industries.",
     "market_opportunity", ["AI", "Finance"], ["Revenue"], "market_driver"),
    ("Launch a B2B2C loyalty coalition platform allowing retail partners to share reward currencies.",
     "market_opportunity", ["Data", "Cloud"], ["Revenue", "CustomerExperience"], "market_driver"),
    ("Create an industry benchmark data product from anonymized platform usage data for subscription revenue.",
     "market_opportunity", ["Data", "AI"], ["Revenue"], "market_driver"),
    ("Develop a cold-chain supply chain visibility SaaS product for the food and beverage segment.",
     "market_opportunity", ["Operations", "Data"], ["Revenue"], "market_driver"),
    ("Launch a real-time treasury management API for embedded-finance integration with ERP vendors.",
     "market_opportunity", ["Finance", "Cloud"], ["Revenue"], "market_driver"),
    ("Build a smart-building energy management platform targeting commercial real estate operators.",
     "market_opportunity", ["Operations", "AI"], ["Revenue", "Cost"], "market_driver"),
    ("Create an AI-powered regulatory change management service for banks, insurers, and asset managers.",
     "market_opportunity", ["AI", "Finance"], ["Revenue"], "market_driver"),

    # ── Operations deep-dive ────────────────────────────────────────────────
    ("Implement a digital quality gate in manufacturing to catch defects before final assembly packaging.",
     "risk_mitigation", ["Operations", "AI"], ["Risk", "Cost"], "operational_problem"),
    ("Build a returns portal with AI-driven resale and refurbishment routing to recover product value.",
     "cost_reduction", ["Operations", "AI"], ["Cost", "Revenue"], "operational_problem"),
    ("Automate production planning with a constraint-based scheduling engine linked to live demand signals.",
     "cost_reduction", ["Operations", "AI"], ["Cost"], "operational_problem"),
    ("Implement a field service management platform to optimize technician dispatch and first-fix rate.",
     "cx_improvement", ["Operations", "Cloud"], ["CustomerExperience", "Cost"], "client_request"),
    ("Deploy a track-and-trace solution to give full cold-chain visibility from supplier to store shelf.",
     "risk_mitigation", ["Operations", "Data"], ["Risk", "CustomerExperience"], "market_driver"),

    # ── Data platform depth ─────────────────────────────────────────────────
    ("Deploy a feature store to accelerate ML model development and avoid redundant pipeline work.",
     "cost_reduction", ["Data", "AI"], ["Cost"], "operational_problem"),
    ("Build a real-time event streaming platform to replace nightly batch ETL and reduce reporting lag.",
     "cost_reduction", ["Data", "Cloud"], ["Cost"], "operational_problem"),
    ("Implement data mesh architecture to decentralize ownership and enforce domain-level data SLAs.",
     "cost_reduction", ["Data", "Operations"], ["Cost", "Risk"], "operational_problem"),
    ("Deploy a data observability platform to monitor pipeline health and catch quality issues before downstream use.",
     "risk_mitigation", ["Data", "Operations"], ["Risk", "Cost"], "operational_problem"),
    ("Launch a customer data platform to unify identities across web, mobile, and CRM for personalized marketing.",
     "market_opportunity", ["Data", "AI"], ["Revenue", "CustomerExperience"], "market_driver"),

    # ── IA depth ────────────────────────────────────────────────────────────
    ("Deploy an LLM-based contract extraction tool to automate clause-level review and risk flagging.",
     "cost_reduction", ["AI", "Operations"], ["Cost", "Risk"], "operational_problem"),
    ("Build a computer vision quality inspection system to detect manufacturing defects at line speed.",
     "risk_mitigation", ["AI", "Operations"], ["Risk", "Cost"], "operational_problem"),
    ("Implement a generative AI copilot for knowledge workers to cut report drafting time by 60%.",
     "cost_reduction", ["AI", "Operations"], ["Cost"], "operational_problem"),
    ("Deploy a reinforcement learning agent for dynamic inventory replenishment to optimize working capital.",
     "cost_reduction", ["AI", "Operations"], ["Cost"], "market_driver"),
    ("Launch a multimodal AI pipeline combining OCR, NLP, and classification for high-volume document intake.",
     "cost_reduction", ["AI", "Data"], ["Cost"], "operational_problem"),

    # ── Autre / strategic ───────────────────────────────────────────────────
    ("Establish a digital center of excellence to accelerate capability building and reduce external consulting spend.",
     "cost_reduction", ["Other", "Operations"], ["Cost", "Revenue"], "market_driver"),
    ("Launch a merger integration office to consolidate systems and processes post-acquisition within 18 months.",
     "risk_mitigation", ["Other", "Operations"], ["Risk", "Cost"], "operational_problem"),
    ("Build an ecosystem partnership program with technology alliances to accelerate co-sell revenue.",
     "market_opportunity", ["Other", "Cloud"], ["Revenue"], "market_driver"),
    ("Implement a portfolio investment scoring model to rank strategic initiatives by NPV and strategic fit.",
     "market_opportunity", ["Other", "Data"], ["Revenue", "Cost"], "market_driver"),
    ("Create a digital twin of the organization to simulate restructuring scenarios before execution.",
     "market_opportunity", ["Other", "AI"], ["Revenue", "Cost"], "market_driver"),

    # ── Autre boost (governance, PMO, strategy, architecture) ──────────────────
    ("Establish a Center of Excellence for digital transformation governance and methodology standardization.",
     "cx_improvement", ["Other"], ["CustomerExperience"], "market_driver"),
    ("Create an API-first integration layer to enable partner ecosystem connectivity and third-party extensibility.",
     "market_opportunity", ["Other", "Cloud"], ["Revenue"], "market_driver"),
    ("Launch a formal enterprise architecture board to govern technology standards and reduce duplication.",
     "cost_reduction", ["Other", "Operations"], ["Cost", "Risk"], "operational_problem"),
    ("Build a strategic vendor management office to consolidate supplier relationships and improve SLA accountability.",
     "cost_reduction", ["Other", "Finance"], ["Cost", "Risk"], "operational_problem"),
    ("Implement a PMO with OKR-based tracking to align project delivery with corporate strategic objectives.",
     "market_opportunity", ["Other"], ["Revenue", "Cost"], "market_driver"),
    ("Create a formal change management practice to reduce adoption failure rates on large transformation programs.",
     "risk_mitigation", ["Other", "HR"], ["Risk", "CustomerExperience"], "operational_problem"),
    ("Develop an innovation lab with time-boxed sprints to prototype and validate new digital business models.",
     "market_opportunity", ["Other", "AI"], ["Revenue"], "market_driver"),
    ("Establish a data-driven board reporting cadence replacing manual slide decks with live executive dashboards.",
     "cost_reduction", ["Other", "Data"], ["Cost"], "operational_problem"),
    ("Launch an outsourcing strategy review to identify which IT services to insource, outsource, or nearshore.",
     "cost_reduction", ["Other", "Operations"], ["Cost"], "market_driver"),
    ("Build a capability maturity assessment framework to benchmark digital readiness across all business units.",
     "market_opportunity", ["Other"], ["Revenue", "Cost"], "market_driver"),
    ("Implement a total cost of ownership model for technology platforms to guide build-buy-partner decisions.",
     "cost_reduction", ["Other", "Finance"], ["Cost"], "operational_problem"),
    ("Create a shared services center for finance, HR, and IT to reduce duplication and improve quality.",
     "cost_reduction", ["Other", "Finance"], ["Cost"], "market_driver"),
    ("Launch a digital skills academy to build internal AI and cloud capabilities and cut external hiring dependency.",
     "cx_improvement", ["Other", "HR"], ["CustomerExperience", "Cost"], "market_driver"),
    ("Establish a cross-functional product board to prioritize the roadmap by business value rather than technical backlog.",
     "market_opportunity", ["Other"], ["Revenue"], "market_driver"),
    ("Implement a technology radar to track emerging tech and guide strategic adoption decisions each quarter.",
     "market_opportunity", ["Other", "AI"], ["Revenue"], "market_driver"),
    ("Build a knowledge management platform to capture institutional expertise before critical retirements.",
     "risk_mitigation", ["Other", "HR"], ["Risk", "Cost"], "operational_problem"),
    ("Design an operating model for the digital business unit clarifying roles, funding, and governance.",
     "cx_improvement", ["Other"], ["CustomerExperience", "Revenue"], "market_driver"),
    ("Create a strategic partnership with a hyperscaler to access co-innovation funds and accelerate cloud adoption.",
     "market_opportunity", ["Other", "Cloud"], ["Revenue", "Cost"], "market_driver"),
    ("Launch a post-merger IT integration playbook to cut integration time from 24 months to 12 months.",
     "cost_reduction", ["Other", "Operations"], ["Cost", "Risk"], "operational_problem"),
    ("Implement a formal technology debt register and quarterly paydown program to reduce systemic operational risk.",
     "risk_mitigation", ["Other", "Operations"], ["Risk", "Cost"], "operational_problem"),
    ("Create an internal venture studio to spin off high-potential digital products as standalone revenue streams.",
     "market_opportunity", ["Other"], ["Revenue"], "market_driver"),
    ("Standardize project delivery methodology across all units to improve predictability and reduce cost overruns.",
     "cost_reduction", ["Other", "Operations"], ["Cost", "CustomerExperience"], "operational_problem"),
    ("Develop a sustainability strategy with measurable KPIs linked to executive compensation and ESG reporting.",
     "risk_mitigation", ["Other"], ["Risk"], "market_driver"),
    ("Build a centralized innovation pipeline from idea submission to board approval with transparent scoring.",
     "market_opportunity", ["Other"], ["Revenue"], "market_driver"),
    ("Implement an IT governance framework to align technology spending with strategic business outcomes.",
     "cost_reduction", ["Other", "Operations"], ["Cost", "Risk"], "market_driver"),
    ("Create a structured hypercare model for post-go-live stabilization to reduce production incidents after deployments.",
     "risk_mitigation", ["Other", "Operations"], ["Risk", "CustomerExperience"], "operational_problem"),
    ("Establish a reverse mentoring program pairing executives with digital-native staff to accelerate culture shift.",
     "cx_improvement", ["Other", "HR"], ["CustomerExperience"], "client_request"),
    ("Build a FinTech partnerships program to extend the core product with best-of-breed third-party capabilities.",
     "market_opportunity", ["Other", "Finance"], ["Revenue"], "market_driver"),
    ("Launch a cross-industry open standards consortium to reduce proprietary vendor lock-in across the sector.",
     "risk_mitigation", ["Other", "Cloud"], ["Risk", "Cost"], "market_driver"),
    ("Implement an enterprise-wide OKR framework to cascade strategic goals to team-level execution metrics.",
     "market_opportunity", ["Other"], ["Revenue", "Cost"], "market_driver"),
    ("Deploy a project portfolio management tool to give leadership real-time visibility into delivery health.",
     "cost_reduction", ["Other", "Data"], ["Cost"], "operational_problem"),
    ("Create a digital ethics board to review AI system deployments for bias, fairness, and explainability.",
     "risk_mitigation", ["Other", "AI"], ["Risk"], "market_driver"),
    ("Establish a nearshore delivery center to reduce IT operating costs while maintaining quality and time-zone alignment.",
     "cost_reduction", ["Other", "Operations"], ["Cost"], "market_driver"),

    # ── Cybersecurity depth ─────────────────────────────────────────────────
    ("Implement a cloud-native CNAPP to continuously assess workload security posture across multi-cloud.",
     "risk_mitigation", ["Cybersecurity", "Cloud"], ["Risk"], "market_driver"),
    ("Deploy deception technology (honeypots, honeytokens) to detect lateral movement inside the network early.",
     "risk_mitigation", ["Cybersecurity"], ["Risk"], "operational_problem"),
    ("Automate threat intelligence ingestion and enrichment to reduce analyst triage time by 70%.",
     "risk_mitigation", ["Cybersecurity", "AI"], ["Risk", "Cost"], "operational_problem"),
    ("Establish a red-team-as-a-service program with quarterly adversary simulation exercises.",
     "risk_mitigation", ["Cybersecurity"], ["Risk"], "market_driver"),
    ("Implement a software supply chain security framework covering SBOM, signing, and dependency audit.",
     "risk_mitigation", ["Cybersecurity", "Cloud"], ["Risk"], "market_driver"),

    # ── Finance depth ───────────────────────────────────────────────────────
    ("Build a real-time payment reconciliation engine to close the daily ledger within 2 hours of cut-off.",
     "cost_reduction", ["Finance", "Data"], ["Cost", "Risk"], "operational_problem"),
    ("Automate financial spreading and covenant monitoring for the corporate loan portfolio.",
     "risk_mitigation", ["Finance", "AI"], ["Risk", "Cost"], "operational_problem"),
    ("Deploy a subscription billing platform to support usage-based pricing models for new SaaS lines.",
     "market_opportunity", ["Finance", "Cloud"], ["Revenue"], "market_driver"),
    ("Implement a tax data management hub to streamline country-by-country reporting under Pillar Two.",
     "risk_mitigation", ["Finance", "Operations"], ["Risk"], "market_driver"),
    ("Build a scenario-based stress testing tool for the treasury team to model interest rate and FX shocks.",
     "risk_mitigation", ["Finance", "Data"], ["Risk"], "market_driver"),

    # ── RH depth ────────────────────────────────────────────────────────────
    ("Implement a skills intelligence platform to map workforce capabilities against strategic talent demand.",
     "cx_improvement", ["HR", "Data"], ["CustomerExperience", "Cost"], "market_driver"),
    ("Deploy a workforce planning simulation tool to model headcount scenarios against revenue forecasts.",
     "cost_reduction", ["HR", "Data"], ["Cost"], "operational_problem"),
    ("Automate contractor onboarding and off-boarding to reduce compliance gaps and access over-provisioning.",
     "risk_mitigation", ["HR", "Cybersecurity"], ["Risk", "Cost"], "operational_problem"),
    ("Build a sabbatical and flexible leave management module that integrates with payroll and project planning.",
     "cx_improvement", ["HR", "Cloud"], ["CustomerExperience"], "client_request"),
    ("Launch a diversity and inclusion analytics dashboard to surface pay equity and representation gaps.",
     "risk_mitigation", ["HR", "Data"], ["Risk"], "market_driver"),

    # ── Data domain boost (30 new) ───────────────────────────────────────────
    ("Build a unified semantic layer on top of the data warehouse so business users query the same definitions.",
     "cost_reduction", ["Data", "Operations"], ["Cost"], "operational_problem"),
    ("Implement a data catalog with automated lineage to make data assets discoverable across all departments.",
     "cost_reduction", ["Data", "Operations"], ["Cost", "Risk"], "operational_problem"),
    ("Deploy a master data management solution to harmonize customer and product records across 12 source systems.",
     "cost_reduction", ["Data", "Operations"], ["Cost", "Risk"], "operational_problem"),
    ("Build a real-time customer 360 view ingesting CRM, web, mobile, and support data for personalized engagement.",
     "cx_improvement", ["Data", "AI"], ["CustomerExperience", "Revenue"], "client_request"),
    ("Implement data governance policies with ownership, stewardship, and quality SLAs for all critical data domains.",
     "risk_mitigation", ["Data", "Operations"], ["Risk"], "market_driver"),
    ("Create a self-service analytics environment with governed access tiers so analysts can explore without IT tickets.",
     "cost_reduction", ["Data", "Cloud"], ["Cost", "CustomerExperience"], "operational_problem"),
    ("Migrate the legacy reporting layer from a monolithic BI tool to a modern cloud-native analytics stack.",
     "cost_reduction", ["Data", "Cloud"], ["Cost"], "operational_problem"),
    ("Deploy a data lakehouse architecture unifying structured and unstructured data for ML and BI workloads.",
     "cost_reduction", ["Data", "Cloud"], ["Cost"], "market_driver"),
    ("Implement GDPR-compliant data subject request automation to handle deletion, portability, and access queries.",
     "risk_mitigation", ["Data", "Cybersecurity"], ["Risk"], "market_driver"),
    ("Build a predictive churn model using customer behavioral data to enable proactive retention campaigns.",
     "market_opportunity", ["Data", "AI"], ["Revenue", "CustomerExperience"], "market_driver"),
    ("Deploy an IoT data ingestion platform to collect sensor telemetry from 50,000 connected devices in real time.",
     "market_opportunity", ["Data", "Operations"], ["Revenue"], "market_driver"),
    ("Create an internal data marketplace allowing teams to share and reuse certified datasets for analytics.",
     "cost_reduction", ["Data", "Operations"], ["Cost"], "operational_problem"),
    ("Implement streaming analytics on clickstream data to enable real-time personalization on the digital storefront.",
     "cx_improvement", ["Data", "AI"], ["CustomerExperience", "Revenue"], "client_request"),
    ("Automate regulatory data submissions by building validated pipelines from source systems to regulator APIs.",
     "risk_mitigation", ["Data", "Finance"], ["Risk", "Cost"], "market_driver"),
    ("Build a geospatial analytics capability to optimize store network, delivery routes, and territory planning.",
     "market_opportunity", ["Data", "Operations"], ["Revenue", "Cost"], "market_driver"),
    ("Deploy a dark data discovery tool to identify and classify unstructured content with sensitive information.",
     "risk_mitigation", ["Data", "Cybersecurity"], ["Risk"], "market_driver"),
    ("Create a financial analytics platform integrating GL, cost center, and project data for real-time P&L visibility.",
     "cost_reduction", ["Data", "Finance"], ["Cost"], "operational_problem"),
    ("Implement synthetic data generation to enable model training and testing without exposing production PII.",
     "risk_mitigation", ["Data", "AI"], ["Risk"], "market_driver"),
    ("Build a data-driven pricing intelligence tool pulling competitor, demand, and margin signals into one view.",
     "market_opportunity", ["Data", "AI"], ["Revenue"], "market_driver"),
    ("Deploy a graph database to model complex relationships between customers, products, and partners for risk scoring.",
     "risk_mitigation", ["Data", "Finance"], ["Risk"], "market_driver"),
    ("Establish a data product mindset with SLAs, versioning, and consumer feedback loops for internal datasets.",
     "cost_reduction", ["Data", "Operations"], ["Cost"], "operational_problem"),
    ("Build a multi-touch attribution model across digital channels to optimize marketing budget allocation.",
     "market_opportunity", ["Data", "AI"], ["Revenue", "Cost"], "market_driver"),
    ("Implement a data quality scorecard published weekly to data owners with automated alerting on degradation.",
     "risk_mitigation", ["Data", "Operations"], ["Risk", "Cost"], "operational_problem"),
    ("Create a competitive intelligence data feed aggregating news, filings, and job postings into analyst dashboards.",
     "market_opportunity", ["Data", "AI"], ["Revenue"], "market_driver"),
    ("Deploy an operational data store providing sub-second query access for call center and branch staff.",
     "cx_improvement", ["Data", "Operations"], ["CustomerExperience", "Cost"], "client_request"),
    ("Build a supply chain control tower aggregating supplier, logistics, and inventory data into one live view.",
     "risk_mitigation", ["Data", "Operations"], ["Risk", "Cost"], "operational_problem"),
    ("Implement a patient outcome analytics platform to identify treatment patterns that reduce readmission rates.",
     "market_opportunity", ["Data", "AI"], ["Revenue", "CustomerExperience"], "market_driver"),
    ("Deploy a real-time fraud scoring service on the payments ledger using behavioral and velocity features.",
     "risk_mitigation", ["Data", "AI"], ["Risk"], "market_driver"),
    ("Create a data literacy training program to raise analytical skills across 2,000 non-technical employees.",
     "cx_improvement", ["Data", "HR"], ["CustomerExperience", "Cost"], "market_driver"),
    ("Build a unified events platform to replace point-to-point integrations with a central publish-subscribe bus.",
     "cost_reduction", ["Data", "Cloud"], ["Cost", "Risk"], "operational_problem"),

    # ── IA domain boost (30 new) ─────────────────────────────────────────────
    ("Deploy a large language model assistant for internal knowledge search across policies, contracts, and wikis.",
     "cost_reduction", ["AI", "Operations"], ["Cost"], "operational_problem"),
    ("Build a generative AI tool that drafts RFP responses from past proposals reducing bid preparation time by 70%.",
     "cost_reduction", ["AI", "Operations"], ["Cost", "Revenue"], "operational_problem"),
    ("Implement an AI-driven predictive maintenance model on HVAC and electrical assets to prevent failures.",
     "cost_reduction", ["AI", "Operations"], ["Cost", "Risk"], "operational_problem"),
    ("Deploy a computer vision system on the production line to detect surface defects at 99.5% accuracy.",
     "risk_mitigation", ["AI", "Operations"], ["Risk", "Cost"], "operational_problem"),
    ("Build an NLP pipeline to classify and route 50,000 inbound customer emails per day without human triage.",
     "cost_reduction", ["AI", "Operations"], ["Cost", "CustomerExperience"], "operational_problem"),
    ("Implement a recommendation engine that personalizes the product catalog for each B2B buyer segment.",
     "market_opportunity", ["AI", "Data"], ["Revenue", "CustomerExperience"], "market_driver"),
    ("Deploy an AI-powered pricing optimizer that adjusts margins dynamically based on demand and competitor signals.",
     "market_opportunity", ["AI", "Finance"], ["Revenue"], "market_driver"),
    ("Build an MLOps platform to standardize model deployment, versioning, monitoring, and retraining workflows.",
     "cost_reduction", ["AI", "Cloud"], ["Cost", "Risk"], "operational_problem"),
    ("Implement a conversational AI agent that handles password resets, access requests, and IT FAQs autonomously.",
     "cost_reduction", ["AI", "Operations"], ["Cost"], "operational_problem"),
    ("Deploy a document intelligence system that extracts structured data from invoices, contracts, and forms at scale.",
     "cost_reduction", ["AI", "Finance"], ["Cost"], "operational_problem"),
    ("Build a credit underwriting model using alternative data signals to expand lending to thin-file customers.",
     "market_opportunity", ["AI", "Finance"], ["Revenue"], "market_driver"),
    ("Implement AI-assisted code review to detect security vulnerabilities before they reach production.",
     "risk_mitigation", ["AI", "Cybersecurity"], ["Risk"], "operational_problem"),
    ("Deploy a demand forecasting model across 200,000 SKUs to reduce overstock by 30% and improve fill rates.",
     "cost_reduction", ["AI", "Operations"], ["Cost", "Revenue"], "operational_problem"),
    ("Build a generative AI co-pilot for financial analysts to automate variance commentary and board pack drafting.",
     "cost_reduction", ["AI", "Finance"], ["Cost"], "operational_problem"),
    ("Implement a next-best-action engine in the CRM to guide sales reps toward the highest-conversion opportunities.",
     "market_opportunity", ["AI", "Operations"], ["Revenue"], "client_request"),
    ("Deploy a voice analytics platform on call recordings to detect compliance breaches and coaching opportunities.",
     "risk_mitigation", ["AI", "Operations"], ["Risk", "CustomerExperience"], "market_driver"),
    ("Build an AI model that predicts employee flight risk 90 days in advance using engagement and performance signals.",
     "cost_reduction", ["AI", "HR"], ["Cost", "Risk"], "operational_problem"),
    ("Implement a real-time anomaly detection model on network traffic to identify zero-day threats within seconds.",
     "risk_mitigation", ["AI", "Cybersecurity"], ["Risk"], "operational_problem"),
    ("Deploy a generative AI assistant that helps procurement teams draft supplier negotiation briefs and clauses.",
     "cost_reduction", ["AI", "Finance"], ["Cost"], "operational_problem"),
    ("Build a multimodal AI system that classifies product images and descriptions for automated catalog enrichment.",
     "cost_reduction", ["AI", "Operations"], ["Cost", "CustomerExperience"], "operational_problem"),
    ("Implement AI-driven logistics route optimization to reduce last-mile delivery costs by 20%.",
     "cost_reduction", ["AI", "Operations"], ["Cost"], "operational_problem"),
    ("Deploy a generative design tool for engineers to generate optimized component geometries and reduce material use.",
     "market_opportunity", ["AI", "Operations"], ["Revenue", "Cost"], "market_driver"),
    ("Build a clinical decision support model that flags drug interactions and dosing anomalies at prescribing time.",
     "risk_mitigation", ["AI", "Operations"], ["Risk", "CustomerExperience"], "market_driver"),
    ("Implement a social listening AI to track brand mentions and surface emerging reputational risks in real time.",
     "risk_mitigation", ["AI", "Data"], ["Risk"], "market_driver"),
    ("Deploy an AI tutor for personalized employee learning paths that adapts content based on quiz performance.",
     "cx_improvement", ["AI", "HR"], ["CustomerExperience"], "client_request"),
    ("Build a call volume forecasting model to optimize contact center staffing and reduce wait times.",
     "cost_reduction", ["AI", "Operations"], ["Cost", "CustomerExperience"], "operational_problem"),
    ("Implement an AI governance framework with model cards, risk tiers, and a mandatory human review gate.",
     "risk_mitigation", ["AI", "Other"], ["Risk"], "market_driver"),
    ("Deploy a satellite imagery AI to monitor agricultural fields and generate crop yield forecasts for commodity trading.",
     "market_opportunity", ["AI", "Data"], ["Revenue"], "market_driver"),
    ("Build an AI-assisted legal research tool that summarizes case law and identifies relevant precedents instantly.",
     "cost_reduction", ["AI", "Finance"], ["Cost"], "market_driver"),
    ("Implement a generative AI system for customer service that drafts empathetic responses in brand voice.",
     "cx_improvement", ["AI", "Operations"], ["CustomerExperience", "Cost"], "client_request"),

    # ── Cloud domain boost (20 new) ──────────────────────────────────────────
    ("Implement a cloud-native CI/CD pipeline to reduce deployment lead time from 2 weeks to same-day.",
     "cost_reduction", ["Cloud", "Operations"], ["Cost", "CustomerExperience"], "operational_problem"),
    ("Migrate the monolithic e-commerce application to microservices on Kubernetes for independent scaling.",
     "market_opportunity", ["Cloud", "Operations"], ["Revenue", "Cost"], "market_driver"),
    ("Deploy a multi-cloud strategy to avoid single-vendor lock-in and improve price negotiation leverage.",
     "risk_mitigation", ["Cloud", "Finance"], ["Risk", "Cost"], "market_driver"),
    ("Implement infrastructure-as-code with policy guardrails to prevent misconfiguration in cloud environments.",
     "risk_mitigation", ["Cloud", "Cybersecurity"], ["Risk"], "operational_problem"),
    ("Build a cloud cost governance dashboard with per-team showback and automated anomaly alerts.",
     "cost_reduction", ["Cloud", "Finance"], ["Cost"], "operational_problem"),
    ("Replatform the data warehouse to a cloud-native MPP engine to reduce query times by 10x.",
     "cost_reduction", ["Cloud", "Data"], ["Cost"], "operational_problem"),
    ("Deploy serverless functions to replace always-on batch VMs and cut compute costs for intermittent workloads.",
     "cost_reduction", ["Cloud", "Operations"], ["Cost"], "operational_problem"),
    ("Build a cloud-native SaaS edition of the on-premise product to address SMB buyers who cannot self-host.",
     "market_opportunity", ["Cloud", "Operations"], ["Revenue"], "market_driver"),
    ("Implement a cloud identity and access management overhaul to enforce least-privilege across 3,000 resources.",
     "risk_mitigation", ["Cloud", "Cybersecurity"], ["Risk"], "operational_problem"),
    ("Deploy a global CDN and edge computing layer to reduce API latency for international users below 100ms.",
     "cx_improvement", ["Cloud", "Operations"], ["CustomerExperience", "Revenue"], "client_request"),
    ("Migrate legacy batch ETL jobs to a cloud-native streaming pipeline to eliminate overnight processing windows.",
     "cost_reduction", ["Cloud", "Data"], ["Cost"], "operational_problem"),
    ("Implement blue-green deployment with automated rollback to eliminate planned maintenance windows.",
     "cx_improvement", ["Cloud", "Operations"], ["CustomerExperience", "Risk"], "client_request"),
    ("Build a platform engineering team with golden paths and internal developer portals to reduce onboarding time.",
     "cost_reduction", ["Cloud", "Operations"], ["Cost", "CustomerExperience"], "operational_problem"),
    ("Deploy a cloud-based PBX and unified communications platform to replace aging on-premise telephony.",
     "cost_reduction", ["Cloud", "Operations"], ["Cost"], "operational_problem"),
    ("Implement automatic cloud resource scheduling to power down non-production environments outside business hours.",
     "cost_reduction", ["Cloud", "Finance"], ["Cost"], "operational_problem"),
    ("Build a hybrid cloud connectivity layer with SD-WAN to unify branch, data center, and cloud networking.",
     "cost_reduction", ["Cloud", "Operations"], ["Cost", "Risk"], "market_driver"),
    ("Deploy a cloud-native observability stack with distributed tracing to cut mean-time-to-resolution by 60%.",
     "cx_improvement", ["Cloud", "Operations"], ["CustomerExperience", "Cost"], "operational_problem"),
    ("Implement a database-as-a-service model to replace self-managed databases and reduce DBA operational burden.",
     "cost_reduction", ["Cloud", "Operations"], ["Cost"], "operational_problem"),
    ("Build a cloud sandbox environment for developers to experiment safely without production access or data risk.",
     "risk_mitigation", ["Cloud", "Cybersecurity"], ["Risk", "CustomerExperience"], "operational_problem"),
    ("Deploy an event-driven architecture on a managed cloud broker to decouple 30 tightly-coupled microservices.",
     "cost_reduction", ["Cloud", "Operations"], ["Cost", "Risk"], "operational_problem"),
]
# fmt: on

WEAK_DOMAIN = {"Cybersecurity", "HR", "Finance"}
WEAK_ORIGIN = {"client_request"}

WORD_SWAPS = [
    ("Deploy ", "Roll out "),
    ("Implement ", "Introduce "),
    ("Automate ", "Set up automation for "),
    ("Launch ", "Start "),
    ("Build ", "Establish "),
    ("Reduce ", "Cut "),
    ("Improve ", "Enhance "),
    ("Develop ", "Create "),
]

PREFIXES = [
    "Business priority: ",
    "We need to ",
    "The steering committee approved ",
    "As part of the 2026 roadmap, ",
]

SUFFIXES = [
    " Target delivery within 12 months.",
    " Requested by business stakeholders.",
    " Part of the digital transformation program.",
    " Expected ROI within 18 months.",
]


def _boost_weight(domains, origin):
    w = 1.0
    if WEAK_DOMAIN.intersection(domains):
        w *= 3.0
    if origin in WEAK_ORIGIN:
        w *= 2.0
    return w


def paraphrase(text):
    out = text
    if random.random() < 0.5:
        for old, new in WORD_SWAPS:
            if out.startswith(old):
                out = new + out[len(old):]
                break
    if random.random() < 0.35:
        out = random.choice(PREFIXES) + out[0].lower() + out[1:]
    if random.random() < 0.35:
        out = out.rstrip(".") + random.choice(SUFFIXES)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _entry(text, obj, domains, impacts, orig, source):
    return {
        "text": text,
        "labels": {
            "objective": obj,
            "domain": domains,
            "impact": impacts,
            "origin": orig,
        },
        "source": source,
    }


def build_dataset(target_n=TARGET_N):
    dataset = []
    seen_texts = set()

    def add(tpl, source):
        text, obj, domains, impacts, orig = tpl
        key = text.strip().lower()
        if key in seen_texts:
            return
        seen_texts.add(key)
        dataset.append(_entry(text, obj, domains, impacts, orig, source))

    for tpl in REAL_PITCHES:
        add(tpl, "real")
    for tpl in EDGE_CASE_PITCHES:
        add(tpl, "edge")
    for tpl in TEMPLATES:
        add(tpl, "synthetic")

    pool = REAL_PITCHES + EDGE_CASE_PITCHES + TEMPLATES
    weights = [_boost_weight(d, o) for _, _, d, _, o in pool]

    attempts = 0
    while len(dataset) < target_n and attempts < target_n * 5:
        attempts += 1
        tpl = random.choices(pool, weights=weights, k=1)[0]
        text, obj, domains, impacts, orig = tpl
        variant = paraphrase(text)
        key = variant.strip().lower()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        dataset.append(_entry(variant, obj, domains, impacts, orig, "paraphrase"))

    random.shuffle(dataset)
    for i, d in enumerate(dataset):
        d["id"] = i
    return dataset[:target_n]


def print_stats(dataset):
    print(f"Dataset: {len(dataset)} pitches\n")

    src = Counter(d.get("source", "?") for d in dataset)
    print("Source mix:", dict(src))

    obj_counts = Counter(d["labels"]["objective"] for d in dataset)
    orig_counts = Counter(d["labels"]["origin"] for d in dataset)
    print("Objective distribution:", dict(obj_counts))
    print("Origin distribution:", dict(orig_counts))

    dom_flat = [v for d in dataset for v in d["labels"]["domain"]]
    print("Domain distribution:", dict(Counter(dom_flat)))


