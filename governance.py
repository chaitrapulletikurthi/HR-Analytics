from __future__ import annotations
import json, hashlib
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List

GENDER_CODED = ["rockstar", "ninja", "dominant", "aggressive", "competitive"]
AGE_CODED = ["young", "digital native", "recent graduate", "energetic"]
EXCLUSIONARY = ["native english", "cultural fit", "able-bodied"]

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def md5(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()

def bias_scan_text(text: str) -> dict:
    t = (text or "").lower()
    hits = {
        "gender_coded": [w for w in GENDER_CODED if w in t],
        "age_coded": [w for w in AGE_CODED if w in t],
        "exclusionary": [w for w in EXCLUSIONARY if w in t],
    }
    n = sum(len(v) for v in hits.values())
    severity = "none" if n == 0 else ("low" if n <= 2 else ("medium" if n <= 5 else "high"))
    return {
        "severity": severity,
        "hits": hits,
        "recommendation": "Rewrite flagged terms with neutral alternatives; review for EEOC/protected-class proxies."
    }

def ensure_dirs(run_dir: Path) -> dict:
    out = run_dir
    gov = run_dir / "governance"
    out.mkdir(parents=True, exist_ok=True)
    gov.mkdir(parents=True, exist_ok=True)
    return {"run_dir": out, "gov_dir": gov}

def write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)

def build_privacy_report(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "timestamp_utc": now_iso(),
        "implemented_controls": [
            "PII redaction before LLM processing (optional toggle)",
            "No raw resume/JD text persisted in artifacts by default (only hashes + structured outputs)",
            "Decision logs for traceability"
        ],
        "planned_controls": [
            "Role-based access control (IAM)",
            "Encryption at rest for stored artifacts",
            "Centralized monitoring/alerting and retention policies",
            "Human-in-the-loop approvals for high-impact decisions"
        ],
    }

def build_decision_log(
    run_id: str,
    employee: dict,
    top_role: dict,
    score: dict,
    explanation: str,
    plan: dict,
    resume_text: str,
    jd_text: str,
    bias_report: dict
) -> dict:
    return {
        "run_id": run_id,
        "timestamp_utc": now_iso(),
        "employee_id": employee.get("employee_id"),
        "employee_name": employee.get("name"),
        "resume_hash": md5(resume_text),
        "jd_hash": md5(jd_text),
        "top_recommendation": {
            "role_id": top_role.get("role_id"),
            "role_title": top_role.get("role_title"),
            "score": score,
            "missing_must_haves": score.get("missing_must_haves"),
            "bias_report": bias_report,
        },
        "explanation": explanation,
        "upskill_plan": plan,
    }
