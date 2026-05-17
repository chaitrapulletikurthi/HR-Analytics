from __future__ import annotations

from typing import Any, Dict, List, Optional
from openai import OpenAI

from .extract import DEFAULT_MODEL, _safe_json_loads, _responses_text


# =============================================================================
# Shared helpers
# =============================================================================

SYSTEM_GUARDRAILS = """
- Do not mention protected traits.
- Do not reference age, gender, race, nationality, disability, etc.
- Keep tone supportive, manager-like, and specific.
- Do not mention cost in the manager narrative.
- Do not invent skills; use provided strengths/evidence and gaps lists.
""".strip()


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split()).strip()


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items or []:
        k = _norm(str(x))
        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(str(x).strip())
    return out


def _resume_skill_names(resume_ex: Dict[str, Any]) -> List[str]:
    skills = resume_ex.get("skills") or []
    out: List[str] = []
    for s in skills:
        if isinstance(s, dict):
            nm = (s.get("name") or "").strip()
            if nm:
                out.append(nm)
        elif isinstance(s, str):
            if s.strip():
                out.append(s.strip())
    return _dedupe_keep_order(out)


def _skill_evidence_examples(resume_ex: Dict[str, Any], limit: int = 8) -> List[List[str]]:
    """
    Return compact evidence pairs like: [["SQL","Built ..."], ["Power BI","Created ..."]]
    Works even if your ResumeExtraction schema varies.
    """
    examples: List[List[str]] = []
    for s in (resume_ex.get("skills") or [])[:limit]:
        if not isinstance(s, dict):
            continue
        nm = (s.get("name") or "").strip()
        if not nm:
            continue
        ev = s.get("evidence") or s.get("examples") or s.get("proof") or []
        ev1 = ""
        if isinstance(ev, list) and ev:
            ev1 = str(ev[0])
        elif isinstance(ev, str):
            ev1 = ev
        examples.append([nm, ev1])
    return examples


# =============================================================================
# Score schema adapter (your current score_match -> old narrative schema)
# =============================================================================

def adapt_score_for_narrative(score: Dict[str, Any]) -> Dict[str, Any]:
    overlap = (score or {}).get("overlap") or {}
    km = overlap.get("keyword_match") or {}
    kmm = overlap.get("keyword_mismatch") or {}

    missing_required = (
        (score or {}).get("missing_must_haves")
        or kmm.get("missing_must_haves")
        or []
    )
    missing_preferred = (
        kmm.get("missing_preferred")
        or []
    )

    strengths = (
        km.get("matched_must_haves")
        or km.get("matched_keywords")
        or []
    )

    return {
        "missing_required": _dedupe_keep_order([str(x) for x in (missing_required or [])]),
        "missing_preferred": _dedupe_keep_order([str(x) for x in (missing_preferred or [])]),
        "strengths": _dedupe_keep_order([str(x) for x in (strengths or [])]),
    }


def classify_gap_context(
    resume_ex: Dict[str, Any],
    current_jd_ex: Dict[str, Any],
    target_jd_ex: Dict[str, Any],
    missing_required: List[str],
    missing_preferred: List[str],
) -> List[Dict[str, Any]]:
    """
    Simple, stable tagging:
      - Build depth in current role
      - New expectation at next role
      - Preferred differentiator
    """
    res = set([_norm(x) for x in _resume_skill_names(resume_ex)])
    cur_must = set([_norm(x) for x in (current_jd_ex.get("must_have_skills") or []) if str(x).strip()])
    tgt_must = set([_norm(x) for x in (target_jd_ex.get("must_have_skills") or []) if str(x).strip()])

    ctx: List[Dict[str, Any]] = []

    for g in (missing_required or []):
        gl = _norm(g)
        if not gl:
            continue
        if gl in tgt_must and gl in cur_must:
            gap_type = "Build depth in current role"
        elif gl in tgt_must and gl not in cur_must:
            gap_type = "New expectation at next role"
        else:
            gap_type = "New expectation at next role"
        ctx.append({"skill": g, "gap_type": gap_type})

    for g in (missing_preferred or []):
        if str(g).strip():
            ctx.append({"skill": str(g).strip(), "gap_type": "Preferred differentiator"})

    # Keep small (the prompt forces 3–5 anyway)
    return ctx[:8]

# =============================================================================
# Upskill Plan (STRICT JSON plan) — Career-advisor + free-first + time ranges
# =============================================================================

def _plan_prompt(
    *,
    employee_name: str,
    current_role: str,
    target_role: str,
    missing_must: List[str],
    missing_pref: List[str],
    resume_skills: List[str],
    hours_per_week_cap: int,
    budget_cap_usd: int,
) -> str:
    no_gaps = (len(missing_must) == 0 and len(missing_pref) == 0)

    return f"""
Create a realistic 6–12 week INTERNAL upskilling plan as a CAREER ADVISOR for a tech professional.

Return STRICT JSON only with schema:
{{
  "role_title": str,
  "total_time_hours": {{"min": int, "max": int}},
  "total_cost_usd": int,
  "assumptions": [str],
  "plan": [
    {{
      "week_range": "1-2",
      "targets_skills": [str],
      "goal": str,
      "activities": [str],
      "time_hours_estimate": {{"min": int, "max": int}},
      "learning_resources": [str],
      "cost_estimate_usd": int,
      "proof_of_skill": str
    }}
  ]
}}

Employee: {employee_name}
Current role: {current_role}
Target role: {target_role}

Missing MUST-HAVES (highest priority): {missing_must}
Missing preferred (secondary): {missing_pref}

Resume skills (do not invent): {resume_skills}

Hard Rules:
1) Skill scope
- targets_skills MUST be a subset of (Missing MUST-HAVES + Missing preferred).
- Do NOT introduce new skills.

2) Priority allocation
- If Missing MUST-HAVES is non-empty: >=70% of total time must address MUST-HAVES.

3) No-gaps mode
- If Missing MUST-HAVES is empty AND Missing preferred is empty:
  - Return a "ready-now" plan with ONLY 1–2 steps.
  - Focus on polish: documentation, stakeholder-ready artifacts, portfolio improvements.
  - Do NOT invent gaps (no ML/Cloud/ETL unless explicitly in missing lists).

4) Cost governance (cert-only spend)
- Learning is assumed FREE via internal AI, official docs, and open resources.
- Do NOT assign costs to Coursera/DataCamp/LinkedIn Learning subscriptions.
- cost_estimate_usd MUST be > 0 ONLY when proof_of_skill explicitly requires a paid certification/exam.
- budget_cap_usd applies ONLY to certification/exam fees (NOT learning platforms).
- Total out-of-pocket cost (cert/exam fees) must be <= {budget_cap_usd}.

Time Rules (IMPORTANT: prevent static hours):
- Use effort RANGES per step: time_hours_estimate = {{"min": X, "max": Y}} with min <= max.
- Do NOT use the same range for every step.
- At least ONE step must be in the 4–8 hour range.
- At most ONE step may be as large as 8–16 hours (unless proof_of_skill is a certification/exam).
- Choose ranges based on step type:
  * Communication / presentation / write-up tasks: 4–8 hours
  * Concept refresh + guided practice: 6–12 hours
  * Hands-on build / implementation tasks: 8–16 hours (max one step unless cert/exam)
- Keep the plan within <= {hours_per_week_cap} hrs/week (assume 6–12 weeks total).
- Keep plan steps to 2–4 items total.

Resources Rules:
- learning_resources must be FREE-FIRST and specific:
  - Official docs (vendor/product docs)
  - YouTube (official/high-quality channels)
  - Free labs / free tier / sample datasets
  - Internal AI-guided practice prompts
- Provide 2–5 resources per step.

Output Rules:
- STRICT JSON only. No markdown. No extra text.

{"NOTE: There are no gaps. Keep it minimal." if no_gaps else ""}
""".strip()



def generate_upskill_plan(
    client: OpenAI,
    employee: dict,
    resume_ex: dict,
    current_jd_ex: dict,
    target_jd_ex: dict,
    score: dict,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Career-advisor plan:
      - time as ranges (min/max)
      - free-first resources
      - cost only when proof implies exam/cert
      - totals recomputed deterministically
    """
    adapted = adapt_score_for_narrative(score)
    missing_must = (adapted.get("missing_required") or [])[:10]
    missing_pref = (adapted.get("missing_preferred") or [])[:8]

    role_title = target_jd_ex.get("role_title") or target_jd_ex.get("display_title") or "Target role"

    hours_per_week_cap = 8
    budget_cap_usd = 350

    prompt = _plan_prompt(
        employee_name=employee.get("name") or "Employee",
        current_role=current_jd_ex.get("role_title") or "Current role",
        target_role=role_title,
        missing_must=missing_must,
        missing_pref=missing_pref,
        resume_skills=_resume_skill_names(resume_ex),
        hours_per_week_cap=hours_per_week_cap,
        budget_cap_usd=budget_cap_usd,
    )

    resp = client.responses.create(model=model, input=prompt, temperature=0)
    raw = _responses_text(resp).strip()
    data = _safe_json_loads(raw) if raw else {}

    # -------------------------
    # Normalize top-level
    # -------------------------
    if not isinstance(data, dict):
        data = {}

    data.setdefault("role_title", role_title)
    data.setdefault("assumptions", [])
    data.setdefault("plan", [])

    if not isinstance(data["assumptions"], list):
        data["assumptions"] = []
    if not isinstance(data["plan"], list):
        data["plan"] = []

    def _as_int(x, default=0) -> int:
        try:
            return int(round(float(x)))
        except Exception:
            return int(default)

    def _norm_list(v) -> list:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        s = str(v).strip() if v is not None else ""
        return [s] if s else []

    def _normalize_time_range(v) -> dict:
        # expect dict {"min":x,"max":y}, but be backward compatible
        if isinstance(v, dict):
            hmin = _as_int(v.get("min", 0), 0)
            hmax = _as_int(v.get("max", hmin), hmin)
            if hmax < hmin:
                hmax = hmin
            return {"min": hmin, "max": hmax}

        # old numeric
        if isinstance(v, (int, float)):
            h = _as_int(v, 0)
            return {"min": h, "max": h}

        # string fallback like "4-6"
        if isinstance(v, str):
            s = v.strip().lower().replace("hours", "").replace("hrs", "").replace("hr", "")
            if "-" in s:
                parts = [p.strip() for p in s.split("-") if p.strip()]
                if len(parts) >= 2:
                    hmin = _as_int(parts[0], 0)
                    hmax = _as_int(parts[1], hmin)
                    if hmax < hmin:
                        hmax = hmin
                    return {"min": hmin, "max": hmax}
            h = _as_int(s, 0)
            return {"min": h, "max": h}

        return {"min": 0, "max": 0}

    def _proof_is_exam_or_cert(proof: str) -> bool:
        p = (proof or "").lower()
        return any(k in p for k in ["exam", "cert", "certification", "certificate", "badge"])

    # -------------------------
    # Normalize each plan step + enforce governance
    # -------------------------
    norm_plan = []
    for s in data["plan"]:
        if not isinstance(s, dict):
            continue

        step = dict(s)
        step.setdefault("week_range", "")
        step.setdefault("goal", "")
        step["targets_skills"] = _norm_list(step.get("targets_skills"))
        step["activities"] = _norm_list(step.get("activities"))
        step["learning_resources"] = _norm_list(step.get("learning_resources"))
        step["proof_of_skill"] = str(step.get("proof_of_skill") or "").strip()

        # time range
        step["time_hours_estimate"] = _normalize_time_range(step.get("time_hours_estimate"))

        # cost governance (0 unless proof implies cert/exam)
        cost = _as_int(step.get("cost_estimate_usd", 0), 0)
        if not _proof_is_exam_or_cert(step["proof_of_skill"]):
            cost = 0
        # keep within your budget cap (optional but consistent with prompt)
        cost = max(0, min(cost, int(budget_cap_usd)))
        step["cost_estimate_usd"] = cost

        # ensure free-first resources exist
        if not step["learning_resources"]:
            step["learning_resources"] = [
                "Official documentation",
                "YouTube (official/high-quality channels)",
                "Free labs / free tier (if applicable)",
                "AI-guided practice and Q&A",
            ]

        norm_plan.append(step)

    data["plan"] = norm_plan

    # -------------------------
    # Recompute totals deterministically (range totals for time)
    # -------------------------
    try:
        tmin = sum(int(st["time_hours_estimate"]["min"]) for st in data["plan"])
        tmax = sum(int(st["time_hours_estimate"]["max"]) for st in data["plan"])
        data["total_time_hours"] = {"min": int(tmin), "max": int(tmax)}
    except Exception:
        data["total_time_hours"] = {"min": 0, "max": 0}

    try:
        data["total_cost_usd"] = int(sum(int(st.get("cost_estimate_usd", 0) or 0) for st in data["plan"]))
    except Exception:
        data["total_cost_usd"] = 0

    # helpful assumptions if LLM forgets
    if not data["assumptions"]:
        data["assumptions"] = [
            "Effort is estimated as a range because learning speed varies by individual.",
            "Free-first learning is assumed using official resources, YouTube, free labs, and internal AI support.",
            "Out-of-pocket cost is included only when proof requires an external certification exam.",
        ]

    return data

# =============================================================================
# Plan summary for HR + Employee
# =============================================================================


def summarize_upskill_plan(plan: dict) -> dict:
    if not plan or not isinstance(plan, dict):
        return {}

    # collect top skills
    skills: List[str] = []
    for s in plan.get("plan", []) or []:
        for t in (s.get("targets_skills", []) or []):
            if t and t not in skills:
                skills.append(t)

    # total_time_hours can be dict {"min","max"} or old int
    tt = plan.get("total_time_hours", {"min": 0, "max": 0})

    if isinstance(tt, dict):
        total_min = int(tt.get("min", 0) or 0)
        total_max = int(tt.get("max", total_min) or total_min)
        if total_max < total_min:
            total_max = total_min
    else:
        total_min = int(tt or 0)
        total_max = int(tt or 0)

    total_cost = int(plan.get("total_cost_usd", 0) or 0)

    # weeks estimate: use midpoint hours / 8 hrs per week cap (or choose max if you want conservative)
    mid_hours = int(round((total_min + total_max) / 2)) if total_max else total_min
    est_weeks = max(1, (mid_hours + 7) // 8)

    return {
        "top_upskill_skills": skills[:5],
        "total_time_hours": {"min": total_min, "max": total_max},
        "estimated_weeks": est_weeks,
        "total_cost_usd": total_cost,
    }

# =============================================================================
# Employee explanation (bullet explanation)
# =============================================================================

def generate_employee_explanation(
    client: OpenAI,
    employee: dict,
    resume_ex: dict,
    target_jd_ex: dict,
    score: dict,
    model: str = DEFAULT_MODEL,
) -> str:
    adapted = adapt_score_for_narrative(score)
    prompt = f"""
Write a concise employee-facing explanation (max 8 bullets):
- Why this role fits
- Matched skills (with evidence)
- Missing MUST-HAVE skills (if any)
- Missing preferred (optional)
- How the plan closes the gap

Employee: {employee.get("name")} ({employee.get("employee_id")})
Role: {target_jd_ex.get("role_title")}

Strengths: {adapted.get("strengths")}
Missing required: {adapted.get("missing_required")}
Missing preferred: {adapted.get("missing_preferred")}

Resume skill evidence examples (use 1–2 specifics): {_skill_evidence_examples(resume_ex, limit=8)}
Return as plain text bullets.
""".strip()

    resp = client.responses.create(model=model, input=prompt, temperature=0.2)
    return _responses_text(resp).strip()


# =============================================================================
# Supportive Manager Narrative (your old style, structured JSON -> formatted text)
# =============================================================================

def _narrative_prompt(
    resume_ex: Dict[str, Any],
    current_jd_ex: Dict[str, Any],
    target_jd_ex: Dict[str, Any],
    adapted_score: Dict[str, Any],
    gap_context: List[Dict[str, Any]],
    tenure_months: Optional[int] = None,
) -> str:
    tenure_line = ""
    if tenure_months is not None:
        tenure_line = f"""
Time in current role (months): {tenure_months}
Tone rule:
- If tenure < 12: emphasize pacing and depth-building
- If tenure >= 12: emphasize transition readiness
Do NOT gate eligibility.
"""

    examples = _skill_evidence_examples(resume_ex, limit=8)

    return f"""{SYSTEM_GUARDRAILS}

You are the employee’s supportive reporting manager.
Write a friendly but powerful development message that feels like a real manager.
No cost mentions. No generic “learn X” language.

Current role: {current_jd_ex.get("role_title") or "Current role"}
Target role: {target_jd_ex.get("role_title") or "Target role"}
{tenure_line}

Employee demonstrated strengths (use these in the summary):
- Strength skills: {adapted_score.get("strengths", [])}
- Evidence examples (use 1–2 specifics): {examples}

Target gaps:
- Missing required: {adapted_score.get("missing_required")}
- Missing preferred: {adapted_score.get("missing_preferred")}

Gap context tags (MUST use these):
{gap_context}

Output requirements:
Return STRICT JSON ONLY with this schema:
{{
  "summary": "4–6 sentences",
  "guidance": [
    {{
      "skill": "string",
      "gap_type": "string (use given tags)",
      "why_it_matters": "string",
      "time_estimate_hours": int,
      "time_estimate_weeks": int,
      "suggested_focus": ["2–4 bullets (work tasks)"],
      "what_ready_looks_like": "manager-verifiable artifact"
    }}
  ],
  "transition_outlook": "2–3 sentences"
}}

Rules:
- guidance items: 3–5 max
- assume 5–7 hrs/week; weeks must match hours roughly
- do not mention cost
""".strip()


def generate_manager_narrative(
    client: OpenAI,
    employee: Dict[str, Any],
    resume_ex: Dict[str, Any],
    current_jd_ex: Dict[str, Any],
    target_jd_ex: Dict[str, Any],
    top3: List[dict],
    score: Dict[str, Any],
    gap: Dict[str, Any],
    plan: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    tenure_months: Optional[int] = None,
) -> str:
    adapted = adapt_score_for_narrative(score)
    gap_ctx = classify_gap_context(
        resume_ex,
        current_jd_ex,
        target_jd_ex,
        adapted.get("missing_required") or [],
        adapted.get("missing_preferred") or [],
    )

    prompt = _narrative_prompt(
        resume_ex,
        current_jd_ex,
        target_jd_ex,
        adapted,
        gap_ctx,
        tenure_months,
    )

    resp = client.responses.create(model=model, input=prompt, temperature=0.35)
    raw = _responses_text(resp).strip()

    # Parse JSON; if it fails, return raw
    try:
        obj = _safe_json_loads(raw)
        summary = (obj.get("summary") or "").strip()
        guidance = obj.get("guidance") or []
        outlook = (obj.get("transition_outlook") or "").strip()

        lines: List[str] = []
        lines.append("📌 Career Summary")
        lines.append("-" * 90)
        lines.append(summary)
        lines.append("")
        lines.append("📈 Upskilling Guidance (Current → Next Role)")
        lines.append("-" * 90)

        for i, item in enumerate(guidance[:5], 1):
            skill = (item.get("skill") or "").strip()
            lines.append(f"{i}. {skill.upper() if skill else 'SKILL'}")
            lines.append(f"   Type     : {item.get('gap_type','')}")
            lines.append(f"   Time     : ~{int(item.get('time_estimate_hours',0) or 0)} hrs ({int(item.get('time_estimate_weeks',0) or 0)} weeks)")
            lines.append("")
            lines.append("   Why this matters")
            lines.append(f"   {item.get('why_it_matters','')}".rstrip())
            lines.append("")
            lines.append("   Suggested focus")
            for b in (item.get("suggested_focus") or [])[:4]:
                lines.append(f"   • {b}")
            lines.append("")
            lines.append('   What "ready" looks like')
            lines.append(f"   {item.get('what_ready_looks_like','')}".rstrip())
            lines.append("")

        lines.append("🎯 Transition Outlook")
        lines.append("-" * 90)
        lines.append(outlook)

        return "\n".join([x for x in lines if x is not None]).strip()

    except Exception:
        return raw


# =============================================================================
# Gap computation (used by engine)
# =============================================================================

def compute_gap(current_jd_ex: dict, target_jd_ex: dict, resume_ex: dict) -> dict:
    def nset(xs):
        return {_norm(x) for x in (xs or []) if str(x).strip()}

    resume_skills = set([_norm(x) for x in _resume_skill_names(resume_ex)])
    cur_must = nset(current_jd_ex.get("must_have_skills"))
    tgt_must = nset(target_jd_ex.get("must_have_skills"))

    return {
        "already_covering_for_target": sorted((resume_skills | cur_must) & tgt_must),
        "missing_for_target": sorted(tgt_must - (resume_skills | cur_must)),
    }
