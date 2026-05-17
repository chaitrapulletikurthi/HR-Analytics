from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI

from .io import load_text_any, redact_pii
from .extract import extract_resume, extract_jd, DEFAULT_MODEL
from .score import score_match
from .governance import (
    ensure_dirs,
    write_json,
    build_privacy_report,
    build_decision_log,
    bias_scan_text,
)
from .plan import (
    generate_upskill_plan,
    summarize_upskill_plan,
    generate_employee_explanation,
    generate_manager_narrative,
    compute_gap,
)


def process_batch(
    employees: List[dict],
    roles: List[dict],
    run_dir: str,
    enable_redaction: bool = True,
    enable_bias_scan: bool = True,
    model: str = DEFAULT_MODEL,
    deep_dive_top_n: int = 1,
    hr_top_n: int = 3,
    score_mode: str = "hybrid",
    semantic_weight: float = 0.40,
    min_confidence: float = 0.55,
) -> dict:

    client = OpenAI()

    run_dir_p = Path(run_dir)
    dirs = ensure_dirs(run_dir_p)
    gov_dir = dirs["gov_dir"]

    run_id = run_dir_p.name.replace("run_", "") if run_dir_p.name.startswith("run_") else run_dir_p.name

    # ---------------------------------------------------------------------
    # 1) Role bank
    # ---------------------------------------------------------------------
    role_bank: List[Dict[str, Any]] = []
    for r in roles:
        jd_path = r["jd_path"]
        role_label = r.get("role_label") or Path(jd_path).stem

        jd_text = load_text_any(jd_path)
        jd_ex = extract_jd(client, jd_text, model=model) or {}

        bias = bias_scan_text(jd_text) if enable_bias_scan else {"severity": "skipped"}

        extracted_title = jd_ex.get("role_title")
        display_title = extracted_title or r.get("role_title") or r.get("role_id") or role_label

        role_bank.append({
            **r,
            "role_label": role_label,
            "jd_text": jd_text,
            "jd_ex": jd_ex,
            "display_title": display_title,
            "bias_report": bias,
        })

    # ---------------------------------------------------------------------
    # 2) Employee extraction cache
    # ---------------------------------------------------------------------
    resume_ex_bank: Dict[str, dict] = {}
    current_jd_ex_bank: Dict[str, dict] = {}
    resume_text_bank: Dict[str, str] = {}

    errors: List[dict] = []
    matches: List[dict] = []

    role_candidate_matrix: Dict[str, List[dict]] = {rb["role_id"]: [] for rb in role_bank}

    for emp in employees:
        emp_id = (emp.get("employee_id") or "").strip()
        emp_name = emp.get("name") or emp_id or "Unknown"

        if not emp_id:
            errors.append({"employee_id": None, "error": "Missing employee_id"})
            continue

        try:
            resume_text = load_text_any(emp["resume_path"])
            if enable_redaction:
                resume_text, _ = redact_pii(resume_text)
            resume_text_bank[emp_id] = resume_text

            resume_ex = extract_resume(client, resume_text, model=model) or {}
            resume_ex_bank[emp_id] = resume_ex

            cur_jd_ex = {"role_title": emp.get("current_role_title") or "Current role"}
            if emp.get("current_jd_path"):
                cur_jd_text = load_text_any(emp["current_jd_path"])
                cur_jd_ex = extract_jd(client, cur_jd_text, model=model) or cur_jd_ex

            current_jd_ex_bank[emp_id] = cur_jd_ex

        except Exception as e:
            errors.append({"employee_id": emp_id, "error": f"Extraction error: {str(e)}"})
            continue

    # ---------------------------------------------------------------------
    # 3) Scoring
    # ---------------------------------------------------------------------
    for emp in employees:
        emp_id = (emp.get("employee_id") or "").strip()
        emp_name = emp.get("name") or emp_id or "Unknown"

        if not emp_id or emp_id not in resume_ex_bank:
            continue

        resume_ex = resume_ex_bank[emp_id]
        current_jd_ex = current_jd_ex_bank.get(emp_id) or {"role_title": "Current role"}

        ranked_roles: List[dict] = []

        for rb in role_bank:
            jd_ex = rb.get("jd_ex") or {}
            sc = score_match(
                resume_ex,
                jd_ex,
                client=client,
                model=model,
                mode=score_mode,
                semantic_weight=semantic_weight,
                min_confidence=min_confidence,
            )

            overlap = sc.get("overlap") or {}
            pct = overlap.get("percent_match") or {}

            ranked_roles.append({
                "role_id": rb["role_id"],
                "role_label": rb["role_label"],
                "role_title": rb["display_title"],
                "score": sc,
                "overlap": overlap,
                "bias_report": rb.get("bias_report") or {"severity": "none"},
                "percent_match": pct,
            })

            role_candidate_matrix[rb["role_id"]].append({
                "employee_id": emp_id,
                "employee_name": emp_name,
                "resume_ex": resume_ex,
                "current_jd_ex": current_jd_ex,
                "score": sc,
                "overlap": overlap,
            })

        ranked_roles.sort(key=lambda x: (x.get("score") or {}).get("overall", 0.0), reverse=True)

        # -----------------------------------------------------------------
        # 4) Deep dive
        # -----------------------------------------------------------------
        deep_dive: List[dict] = []
        top_plan = {}
        manager_note = ""
        employee_expl = ""

        n = max(0, int(deep_dive_top_n or 0))
        top_roles_for_emp = ranked_roles[:n] if n > 0 else []

        for i, top in enumerate(top_roles_for_emp):
            rb = next((x for x in role_bank if x["role_id"] == top["role_id"]), None)
            target_jd_ex = (rb.get("jd_ex") if rb else {}) or {}
            target_title = (rb.get("display_title") if rb else top.get("role_title")) or "Target role"
            target_label = (rb.get("role_label") if rb else top.get("role_label")) or top.get("role_id")

            # ✅ Ensure prompts always see correct title/label
            target_jd_ex = dict(target_jd_ex or {})
            target_jd_ex["role_title"] = target_title
            target_jd_ex["role_label"] = target_label

            gap = compute_gap(current_jd_ex, target_jd_ex, resume_ex)

            # ✅ Make score schema consistent for plan/narrative
            sc_clean = dict(top["score"] or {})
            ov = sc_clean.get("overlap") or {}
            kmm = (ov.get("keyword_mismatch") or {})
            kmm["missing_must_haves"] = sc_clean.get("missing_must_haves") or kmm.get("missing_must_haves") or []
            ov["keyword_mismatch"] = kmm
            sc_clean["overlap"] = ov

            plan = generate_upskill_plan(
                client,
                employee={
                    "employee_id": emp_id,
                    "name": emp_name,
                    "target_role_title": target_title,
                },
                resume_ex=resume_ex,
                current_jd_ex=current_jd_ex,
                target_jd_ex=target_jd_ex,
                score=sc_clean,
                model=model,
            )

            plan_summary = summarize_upskill_plan(plan)

            expl = generate_employee_explanation(
                client,
                employee={"employee_id": emp_id, "name": emp_name},
                resume_ex=resume_ex,
                target_jd_ex=target_jd_ex,
                score=sc_clean,
                model=model,
            )

            note = generate_manager_narrative(
                client,
                employee={"employee_id": emp_id, "name": emp_name},
                resume_ex=resume_ex,
                current_jd_ex=current_jd_ex,
                target_jd_ex=target_jd_ex,
                top3=[
                    {
                        "role_id": x["role_id"],
                        "role_title": x.get("role_title"),
                        "role_label": x.get("role_label"),
                        "percent_match": (x.get("overlap") or {}).get("percent_match"),
                    }
                    for x in ranked_roles[:3]
                ],
                score=sc_clean,
                gap=gap,
                plan=plan,
                model=model,
            )

            deep_dive.append({
                "role_id": top["role_id"],
                "role_label": target_label,
                "role_title": target_title,
                "plan": plan,
                "plan_summary": plan_summary,
                "employee_explanation": expl,
                "manager_narrative": note,
                "gap": gap,
            })

            if i == 0:
                top_plan = plan
                employee_expl = expl
                manager_note = note

                try:
                    decision_log = build_decision_log(
                        run_id=run_id,
                        employee={"employee_id": emp_id, "name": emp_name},
                        top_role={"role_id": top["role_id"], "role_title": target_title, "role_label": target_label},
                        score=sc_clean,
                        explanation=note,
                        plan=plan,
                        resume_text=resume_text_bank.get(emp_id, ""),
                        jd_text=(rb.get("jd_text") if rb else "") or "",
                        bias_report=(rb.get("bias_report") if rb else {"severity": "none"}),
                    )
                    write_json(gov_dir / f"decision_log_{emp_id}_{run_id}.json", decision_log)
                except Exception:
                    pass

        matches.append({
            "employee_id": emp_id,
            "employee_name": emp_name,
            "ranked_roles": ranked_roles,
            "deep_dive": deep_dive,
            "top_plan": top_plan,
            "employee_explanation": employee_expl,
            "manager_narrative": manager_note,
        })

    # ---------------------------------------------------------------------
    # 5) HR role-centric view
    # ---------------------------------------------------------------------
    hr_role_view: Dict[str, Any] = {}

    if hr_top_n and int(hr_top_n) > 0:
        hr_n = int(hr_top_n)

        for rb in role_bank:
            role_id = rb["role_id"]
            candidates = role_candidate_matrix.get(role_id, [])
            candidates.sort(key=lambda x: (x.get("score") or {}).get("overall", 0.0), reverse=True)

            top_candidates = []
            for c in candidates[:hr_n]:
                sc = c.get("score") or {}
                overlap = c.get("overlap") or {}
                pct = overlap.get("percent_match") or {}

                tgt_ex = dict(rb["jd_ex"] or {})
                tgt_ex["role_title"] = rb["display_title"]
                tgt_ex["role_label"] = rb["role_label"]

                # consistency
                sc_clean = dict(sc or {})
                ov = sc_clean.get("overlap") or {}
                kmm = (ov.get("keyword_mismatch") or {})
                kmm["missing_must_haves"] = sc_clean.get("missing_must_haves") or kmm.get("missing_must_haves") or []
                ov["keyword_mismatch"] = kmm
                sc_clean["overlap"] = ov

                plan = generate_upskill_plan(
                    client,
                    employee={
                        "employee_id": c["employee_id"],
                        "name": c["employee_name"],
                        "target_role_title": rb["display_title"],
                    },
                    resume_ex=c["resume_ex"],
                    current_jd_ex=c["current_jd_ex"],
                    target_jd_ex=tgt_ex,
                    score=sc_clean,
                    model=model,
                )
                upskill_summary = summarize_upskill_plan(plan)

                top_candidates.append({
                    "employee_id": c["employee_id"],
                    "employee_name": c["employee_name"],
                    "overall_pct": float(pct.get("overall_pct") or (sc.get("overall", 0.0) * 100)),
                    "must_have_pct": float(pct.get("must_have_pct") or (sc.get("must_have", 0.0) * 100)),
                    "preferred_pct": float(pct.get("preferred_pct") or (sc.get("preferred", 0.0) * 100)),
                    "missing_must_haves": sc_clean.get("missing_must_haves") or [],
                    "matched_must_haves": (overlap.get("keyword_match") or {}).get("matched_must_haves") or [],
                    "bias_severity": (rb.get("bias_report") or {}).get("severity", "none"),
                    "upskill_summary": upskill_summary,
                })

            hr_role_view[role_id] = {
                "role_id": role_id,
                "role_title": rb["display_title"],
                "role_label": rb["role_label"],
                "bias_severity": (rb.get("bias_report") or {}).get("severity", "none"),
                "top_candidates": top_candidates,
            }

    # ---------------------------------------------------------------------
    # 6) Summary table
    # ---------------------------------------------------------------------
    summary_table = []
    for m in matches:
        top_r = (m.get("ranked_roles") or [{}])[0] or {}
        overlap = top_r.get("overlap") or {}
        pct = overlap.get("percent_match") or {}
        miss = (overlap.get("keyword_mismatch") or {}).get("missing_must_haves") or []
        summary_table.append({
            "employee_id": m.get("employee_id"),
            "employee_name": m.get("employee_name"),
            "top_role_title": top_r.get("role_title"),
            "top_role_label": top_r.get("role_label"),
            "overall_pct": pct.get("overall_pct"),
            "must_have_pct": pct.get("must_have_pct"),
            "preferred_pct": pct.get("preferred_pct"),
            "missing_must_haves": ", ".join(miss[:8]),
        })

    # ---------------------------------------------------------------------
    # 7) Governance artifacts
    # ---------------------------------------------------------------------
    privacy = build_privacy_report(run_id)
    artifact_paths = {
        "privacy_report": write_json(gov_dir / f"privacy_report_{run_id}.json", privacy),
        "batch_results": write_json(run_dir_p / f"batch_results_{run_id}.json", {"matches": matches, "errors": errors}),
        "hr_role_view": write_json(run_dir_p / f"hr_role_view_{run_id}.json", hr_role_view),
    }

    return {
        "matches": matches,
        "errors": errors,
        "hr_role_view": hr_role_view,
        "artifact_paths": artifact_paths,
        "summary_table": summary_table,
    }
