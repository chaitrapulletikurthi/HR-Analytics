<div align="center">

# HR Analytics — Talent Mobility Engine

### What if your next best hire is already on your payroll?

*Chaitra Neha Pulletikurthi · MS Business Analytics*

[![View App](https://img.shields.io/badge/View-App-blueviolet?style=for-the-badge)](app.py)
[![Engine](https://img.shields.io/badge/Pipeline-Engine-green?style=for-the-badge)](engine.py)

</div>

---

## The Problem

Most companies spend time and money searching for external candidates — while overlooking the people already inside the organization who could do the job. Internal mobility is hard because matching people to roles manually is slow, inconsistent, and often biased without anyone realizing it.

---

## What This Does

An end-to-end AI system that reads employee resumes and open job descriptions, scores how well each person fits each role, flags bias in job postings, protects personal data, and tells managers exactly who to consider — and what skills each candidate would need to develop to succeed.

---

## How It Works — 6 Stages

**Stage 1 — Read the Job Descriptions**
The system parses each role's requirements and scans the job description for biased language before anything else happens. Words like *"rockstar," "ninja," "young," "digital native,"* or *"native English"* are flagged automatically — because the way a job is written affects who applies.

**Stage 2 — Extract Employee Profiles**
Resumes are loaded and all personally identifiable information (name, contact details) is redacted before the data is sent to any AI model. Privacy first, always.

**Stage 3 — Score Every Employee Against Every Role**
Each employee gets a match score per role using a hybrid approach:

| Signal | Weight |
|--------|--------|
| Must-have skills coverage | 75% |
| Preferred skills coverage | 25% |

Scoring itself blends two methods:
- **Keyword matching** (60%) — fast, literal skill comparison
- **Semantic matching via GPT-4o mini** (40%) — understands context, not just exact words

Minimum confidence threshold of **0.55** applied to semantic matches to prevent false positives.

**Stage 4 — Deep Dive on Top Matches**
For the strongest candidate-role pairs, the system generates a personalized upskilling plan — showing exactly which skills the employee has, which ones are missing, and what they'd need to close the gap.

**Stage 5 — HR View**
Flips the perspective: instead of "which roles fit this employee," it shows "which employees best fit this role" — ranked, with capability breakdowns, ready for a hiring manager to act on.

**Stage 6 — Governance & Audit Trail**
Every decision is logged. Bias reports, skill gap analyses, privacy redaction records, and match rationale are saved as structured JSON — so HR teams can explain every recommendation and stay compliant.

---

## What You Get as Output

- Ranked role matches per employee with detailed scoring
- Top candidate shortlist per open role
- Personalized upskilling plans for internal mobility
- Bias report for every job description
- Full audit trail for compliance

---

## Built-In Fairness Features

The system actively scans for three types of problematic language in job postings:

- **Gender-coded** — "aggressive," "competitive," "dominant"
- **Age-coded** — "young," "recent graduate," "energetic," "digital native"
- **Exclusionary** — "native English," "cultural fit," "able-bodied"

Each finding is assigned a severity level (Low / Medium / High) so HR teams know what to fix and how urgently.

---

## What's in This Repo

| File | What it does |
|------|-------------|
| `app.py` | Entry point — runs the full batch pipeline |
| `engine.py` | Orchestrates all 6 stages end to end |
| `score.py` | Hybrid keyword + semantic scoring logic |
| `extract.py` | Parses resumes and job descriptions |
| `governance.py` | Bias scanning, PII redaction, decision logging |
| `plan.py` | Generates personalized upskilling plans |
| `dashboard_integration.py` | Connects results to a visual dashboard |
| `talent_mobility_dashboard.py` | Talent mobility analytics and reporting |

---

## Built With

`Python` · `OpenAI GPT-4o mini` · `NLP` · `Semantic Matching` · `TF-IDF` · `Pandas` · `HR Analytics` · `Responsible AI`

---

## Skills Demonstrated

`AI Pipeline Design` · `LLM Integration` · `Bias Detection` · `PII Redaction` · `Talent Analytics` · `Scoring Models` · `Governance & Compliance` · `Python`
