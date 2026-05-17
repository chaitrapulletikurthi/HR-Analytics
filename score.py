from __future__ import annotations

from typing import Dict, List, Optional
import re
import json

# -------------------------
# Baseline (keyword) overlap
# -------------------------

_STOP = [
    "strong", "proficiency", "proficient", "experience", "hands on", "hands-on",
    "solid", "advanced", "basic", "knowledge of", "ability to", "familiarity with",
    "working knowledge", "understanding of", "demonstrated", "proven"
]


def _norm_skill(s: str) -> str:
    s = (s or "").strip().lower()
    # keep + and # (c++, c#), remove other punctuation
    s = re.sub(r"[^a-z0-9\s\+\#]", " ", s)
    for p in _STOP:
        s = s.replace(p, " ")
    s = re.sub(r"\s+", " ", s).strip()

    # light canonicalization
    s = s.replace("powerbi", "power bi")
    s = s.replace("a/b", "ab").replace("a b", "ab")
    return s


def _resume_skill_strings(resume_ex: dict) -> List[str]:
    out = []
    for x in (resume_ex.get("skills") or []):
        if isinstance(x, dict):
            name = x.get("name") or ""
            ev = x.get("evidence") or ""
            out.append(name)
            if isinstance(ev, str) and ev.strip():
                out.append(ev)
        elif isinstance(x, str):
            out.append(x)
    return [s for s in out if s and s.strip()]


def _baseline_match(jd_skill: str, resume_norms: List[str]) -> bool:
    """
    Conservative lexical match:
    - normalize both sides
    - substring either direction
    Fixes: "strong proficiency in sql" vs "sql"
    """
    j = _norm_skill(jd_skill)
    if not j:
        return False
    return any((j in r) or (r in j) for r in resume_norms if r)


def compute_skill_overlap_baseline(resume_ex: dict, jd_ex: dict) -> dict:
    resume_norms = [_norm_skill(s) for s in _resume_skill_strings(resume_ex)]
    resume_norms = [s for s in resume_norms if s]

    must = [x for x in (jd_ex.get("must_have_skills") or []) if x and str(x).strip()]
    pref = [x for x in (jd_ex.get("preferred_skills") or []) if x and str(x).strip()]

    must_hit, must_miss = [], []
    for m in must:
        (must_hit if _baseline_match(m, resume_norms) else must_miss).append(_norm_skill(m))

    pref_hit, pref_miss = [], []
    for p in pref:
        (pref_hit if _baseline_match(p, resume_norms) else pref_miss).append(_norm_skill(p))

    must_cov = (len(must_hit) / len(must)) if must else 0.0
    pref_cov = (len(pref_hit) / len(pref)) if pref else 0.0

    # transparent baseline rule
    overall = 0.75 * must_cov + 0.25 * pref_cov

    return {
        "keyword_match": {
            "matched_must_haves": must_hit,
            "matched_preferred": pref_hit,
        },
        "keyword_mismatch": {
            "missing_must_haves": must_miss,
            "missing_preferred": pref_miss,
        },
        "percent_match": {
            "must_have_pct": round(must_cov * 100, 2),
            "preferred_pct": round(pref_cov * 100, 2),
            "overall_pct": round(overall * 100, 2),
        },
    }


# -------------------------
# Semantic (LLM) overlap
# -------------------------

def _safe_json_loads_local(s: str) -> dict:
    """
    Minimal robust JSON loader to protect scoring from malformed model output.
    (Does NOT hardcode business logic; only parses.)
    """
    s = (s or "").strip()

    # strip markdown fences
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1].strip()

    # keep first JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end + 1]

    # remove control chars that break json.loads
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # common fixes: remove trailing commas
        s2 = re.sub(r",\s*}", "}", s)
        s2 = re.sub(r",\s*]", "]", s2)
        return json.loads(s2)


def _responses_text_local(resp) -> str:
    """
    Works across SDK variants:
    - resp.output_text (new SDK)
    - resp.output[0].content[0].text (fallback)
    """
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text

    # fallback: try common nested layout
    try:
        chunks = []
        for item in (resp.output or []):
            for c in (getattr(item, "content", None) or []):
                t = getattr(c, "text", None)
                if t:
                    chunks.append(t)
        return "\n".join(chunks).strip()
    except Exception:
        return str(resp)


def compute_skill_overlap_semantic(
    client,
    resume_ex: dict,
    jd_ex: dict,
    model: str,
    *,
    min_confidence: float = 0.55,
) -> dict:
    """
    Semantic evaluator:
    - For each JD skill, model decides matched/not + confidence + evidence.
    - We convert that to coverage %.
    """
    resume_skills = resume_ex.get("skills") or []
    must = jd_ex.get("must_have_skills") or []
    pref = jd_ex.get("preferred_skills") or []

    prompt = f"""
You are evaluating skill overlap between a resume and a job description.

Return STRICT JSON ONLY with keys:
{{
  "must": [{{"skill": str, "matched": bool, "confidence": float, "evidence": str, "why": str}}],
  "preferred": [{{"skill": str, "matched": bool, "confidence": float, "evidence": str, "why": str}}]
}}

Rules:
- Use ONLY the provided resume skills/evidence. Do NOT invent experience.
- confidence must be between 0.0 and 1.0.
- evidence must be a short phrase copied from resume evidence if available, else "".
- Keep "why" to one short sentence.

Resume skills (with evidence objects/strings):
{resume_skills}

Must-have skills:
{must}

Preferred skills:
{pref}
""".strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
    )

    raw = _responses_text_local(resp)
    data = _safe_json_loads_local(raw)

    must_rows = data.get("must") or []
    pref_rows = data.get("preferred") or []

    def _is_hit(r: dict) -> bool:
        return bool(r.get("matched")) and float(r.get("confidence") or 0.0) >= float(min_confidence)

    must_hit = [r.get("skill") for r in must_rows if _is_hit(r)]
    must_miss = [r.get("skill") for r in must_rows if not _is_hit(r)]
    pref_hit = [r.get("skill") for r in pref_rows if _is_hit(r)]
    pref_miss = [r.get("skill") for r in pref_rows if not _is_hit(r)]

    must_cov = (len(must_hit) / len(must_rows)) if must_rows else 0.0
    pref_cov = (len(pref_hit) / len(pref_rows)) if pref_rows else 0.0
    overall = 0.75 * must_cov + 0.25 * pref_cov

    return {
        "keyword_match": {  # keep key names so engine printing doesn't break
            "matched_must_haves": [(_norm_skill(x) if isinstance(x, str) else x) for x in must_hit if x],
            "matched_preferred": [(_norm_skill(x) if isinstance(x, str) else x) for x in pref_hit if x],
        },
        "keyword_mismatch": {
            "missing_must_haves": [(_norm_skill(x) if isinstance(x, str) else x) for x in must_miss if x],
            "missing_preferred": [(_norm_skill(x) if isinstance(x, str) else x) for x in pref_miss if x],
        },
        "percent_match": {
            "must_have_pct": round(must_cov * 100, 2),
            "preferred_pct": round(pref_cov * 100, 2),
            "overall_pct": round(overall * 100, 2),
        },
        # extra: store semantic evidence for decision log / governance
        "semantic_audit": {
            "must": must_rows,
            "preferred": pref_rows,
            "min_confidence": min_confidence,
        }
    }


# -------------------------
# Hybrid scorer (baseline + semantic)
# -------------------------

def compute_skill_overlap_hybrid(
    client,
    resume_ex: dict,
    jd_ex: dict,
    model: str,
    *,
    semantic_weight: float = 0.40,
    min_confidence: float = 0.55,
) -> dict:
    """
    Hybrid = (1-w)*baseline + w*semantic
    - Keep baseline keyword match/mismatch for transparency.
    - Add semantic audit + semantic percent.
    - Use blended percent_match for ranking.
    """
    baseline = compute_skill_overlap_baseline(resume_ex, jd_ex)
    semantic = compute_skill_overlap_semantic(
        client=client,
        resume_ex=resume_ex,
        jd_ex=jd_ex,
        model=model,
        min_confidence=min_confidence,
    )

    b = baseline["percent_match"]
    s = semantic["percent_match"]

    def blend(k: str) -> float:
        return (1.0 - semantic_weight) * float(b.get(k) or 0.0) + semantic_weight * float(s.get(k) or 0.0)

    # Use baseline keyword lists for printing (judge-friendly transparency)
    overlap = {
        "keyword_match": baseline["keyword_match"],
        "keyword_mismatch": baseline["keyword_mismatch"],
        "percent_match": {
            "must_have_pct": round(blend("must_have_pct"), 2),
            "preferred_pct": round(blend("preferred_pct"), 2),
            "overall_pct": round(blend("overall_pct"), 2),
        },
        "baseline_percent_match": b,
        "semantic_percent_match": s,
        "semantic_audit": semantic.get("semantic_audit"),
        "hybrid": {
            "semantic_weight": semantic_weight,
            "min_confidence": min_confidence,
        }
    }
    return overlap


def reconcile_missing_must_haves(
    keyword_missing: List[str],
    semantic_audit: Optional[dict],
    *,
    min_confidence: float = 0.55,
) -> List[str]:
    """
    Remove from keyword_missing if semantic audit says it's matched with confidence >= threshold.
    Uses _norm_skill + substring matching so:
      "with power bi" ~ "experience with power bi"
    """
    if not keyword_missing or not semantic_audit:
        return keyword_missing or []

    semantic_must = semantic_audit.get("must") or []

    # normalized list of semantic matched skills
    sem_hits: List[str] = []
    for row in semantic_must:
        if bool(row.get("matched")) and float(row.get("confidence") or 0.0) >= float(min_confidence):
            sem_hits.append(_norm_skill(str(row.get("skill") or "")))

    sem_hits = [s for s in sem_hits if s]

    corrected: List[str] = []
    for skill in keyword_missing:
        k = _norm_skill(str(skill or ""))
        if not k:
            continue

        # if semantic hit covers this missing skill, drop it
        covered = any((k in sh) or (sh in k) for sh in sem_hits)
        if not covered:
            corrected.append(skill)

    return corrected


def score_match(
    resume_ex: dict,
    jd_ex: dict,
    *,
    client=None,
    model: str = "gpt-4o-mini",
    mode: str = "hybrid",          # "baseline" or "hybrid"
    semantic_weight: float = 0.40,
    min_confidence: float = 0.55,
) -> dict:
    """
    Backward-compatible score dict used by engine.

    - baseline mode works without LLM client
    - hybrid mode requires client
    """
    if mode == "baseline" or client is None:
        overlap = compute_skill_overlap_baseline(resume_ex, jd_ex)
    else:
        overlap = compute_skill_overlap_hybrid(
            client=client,
            resume_ex=resume_ex,
            jd_ex=jd_ex,
            model=model,
            semantic_weight=semantic_weight,
            min_confidence=min_confidence,
        )

    pct = overlap["percent_match"]

    # Start with keyword-based missing list (baseline transparency)
    missing_must = overlap["keyword_mismatch"]["missing_must_haves"]

    # If hybrid mode has semantic audit, reconcile to avoid "matched + missing" contradictions
    semantic_audit = overlap.get("semantic_audit") if isinstance(overlap, dict) else None
    if semantic_audit:
        missing_must = reconcile_missing_must_haves(
            missing_must,
            semantic_audit,
            min_confidence=min_confidence,
        )

    return {
        "overall": round(pct["overall_pct"] / 100.0, 4),
        "must_have": round(pct["must_have_pct"] / 100.0, 4),
        "preferred": round(pct["preferred_pct"] / 100.0, 4),
        "missing_must_haves": missing_must,
        "matched_must_haves": overlap["keyword_match"]["matched_must_haves"],
        "matched_preferred": overlap["keyword_match"]["matched_preferred"],
        "overlap": overlap,
    }
