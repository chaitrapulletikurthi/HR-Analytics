import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import sys
from typing import List, Tuple
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.talent_agent.engine import process_batch
from src.talent_agent.io import load_text_any


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="EY Talent Mobility Intelligence",
    page_icon="💼",
    layout="wide",
)

# =============================================================================
# EY COLORS
# =============================================================================
EY_YELLOW = "#FFE600"
EY_BLACK = "#2E2E38"
EY_DARK_GRAY = "#747480"
EY_GREEN = "#78BE20"
EY_ORANGE = "#FF6900"
EY_BLUE = "#0077C8"


# =============================================================================
# ENHANCED STYLING (CSS) - WITH MODALS AND INTERACTIVE ELEMENTS
# =============================================================================
st.markdown(
    f"""
    <style>
      /* ---------- Skill pills ---------- */
      .pill {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 12px;
        margin: 4px 6px 4px 0;
        background: {EY_ORANGE};
        color: #FFFFFF;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s;
      }}
      
      .pill:hover {{
        background: {EY_BLUE};
        transform: scale(1.05);
      }}

      /* ---------- Generic card ---------- */
      .card {{
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid #EEE;
        background: #FFFFFF;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 12px;
        transition: all 0.3s ease;
      }}
      
      .card:hover {{
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
      }}
      
      /* ---------- Collapsible sections ---------- */
      .collapsible {{
        cursor: pointer;
        padding: 10px;
        border-radius: 8px;
        background: {EY_YELLOW};
        color: {EY_BLACK};
        margin: 8px 0;
        font-weight: 600;
        transition: background 0.2s;
      }}
      
      .collapsible:hover {{
        background: {EY_ORANGE};
        color: white;
      }}

      /* ---------- Text helpers ---------- */
      .muted {{
        color: {EY_DARK_GRAY};
      }}

      .title {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {EY_BLACK};
      }}
      
      /* ---------- Interactive badges ---------- */
      .badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 4px;
      }}
      
      .badge-success {{
        background: {EY_GREEN};
        color: white;
      }}
      
      .badge-warning {{
        background: {EY_ORANGE};
        color: white;
      }}
      
      .badge-info {{
        background: {EY_BLUE};
        color: white;
      }}
      
      /* ---------- Evidence highlight ---------- */
      .evidence-box {{
        background: #F8F9FA;
        border-left: 4px solid {EY_GREEN};
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
        font-style: italic;
      }}
      
      /* ---------- Semantic audit styling ---------- */
      .semantic-item {{
        background: white;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
      }}
      
      .confidence-high {{
        border-left: 4px solid {EY_GREEN};
      }}
      
      .confidence-medium {{
        border-left: 4px solid {EY_ORANGE};
      }}
      
      .confidence-low {{
        border-left: 4px solid #DC3545;
      }}

      /* =====================================================
         🔥 Employee Welcome Card (EY Yellow)
         ===================================================== */
      div.card.welcome-card {{
        background: {EY_YELLOW} !important;
        color: {EY_BLACK} !important;
        border: none !important;
      }}

      div.card.welcome-card .title {{
        color: {EY_BLACK} !important;
      }}

      div.card.welcome-card .muted {{
        color: {EY_BLACK} !important;
        opacity: 0.85 !important;
      }}

      /* Extra safety: Streamlit markdown wrapper */
      [data-testid="stMarkdownContainer"] div.card.welcome-card {{
        background: {EY_YELLOW} !important;
      }}
      
      /* ---------- Comparison table styling ---------- */
      .comparison-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
      }}
      
      .comparison-table th {{
        background: {EY_BLACK};
        color: white;
        padding: 12px;
        text-align: left;
      }}
      
      .comparison-table td {{
        padding: 10px;
        border-bottom: 1px solid #E0E0E0;
      }}
      
      .comparison-table tr:hover {{
        background: #F8F9FA;
      }}
      
      /* ---------- Loading state ---------- */
      .loading-state {{
        text-align: center;
        padding: 40px;
        color: {EY_DARK_GRAY};
      }}
      
      /* ---------- Interactive button styling ---------- */
      .custom-button {{
        background: {EY_YELLOW};
        color: {EY_BLACK};
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        cursor: pointer;
        font-weight: 600;
        transition: all 0.2s;
      }}
      
      .custom-button:hover {{
        background: {EY_ORANGE};
        color: white;
        transform: translateY(-2px);
      }}
      
      /* ---------- Progress bar ---------- */
      .skill-progress {{
        background: #E0E0E0;
        border-radius: 8px;
        height: 24px;
        position: relative;
        overflow: hidden;
        margin: 8px 0;
      }}
      
      .skill-progress-bar {{
        background: linear-gradient(90deg, {EY_GREEN}, {EY_YELLOW});
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: {EY_BLACK};
        font-weight: 600;
        font-size: 0.85rem;
        transition: width 0.5s ease;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HELPERS
# =============================================================================
def save_upload(uploaded_file, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / uploaded_file.name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def infer_name_from_resume_text(text: str) -> str:
    if not text:
        return ""
    for line in text.splitlines():
        s = (line or "").strip()
        if not s:
            continue
        if 1 < len(s.split()) <= 5 and s.replace(" ", "").isalpha():
            return s
    return ""


def gauge(value, title):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value or 0),
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": EY_YELLOW},
                "steps": [
                    {"range": [0, 50], "color": "#FFE6E6"},
                    {"range": [50, 70], "color": "#FFF4E6"},
                    {"range": [70, 100], "color": "#E6F7E6"},
                ],
            },
        )
    )
    fig.update_layout(height=220, margin=dict(t=40, b=0))
    return fig


def role_display_name(role_title: str, role_label: str, role_id: str) -> str:
    label = role_label or role_id
    return f"{role_title} ({label})"


def _norm_skill(s: str) -> str:
    return " ".join((s or "").lower().split()).strip()


def flatten_plan_targets(plan: dict) -> List[str]:
    """Collect all targets_skills from the plan steps into a single list."""
    targets = []
    for step in (plan or {}).get("plan", []) or []:
        for t in (step.get("targets_skills", []) or []):
            if t and str(t).strip():
                targets.append(str(t).strip())

    # de-dupe while preserving order
    seen = set()
    out = []
    for t in targets:
        k = _norm_skill(t)
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    return out


def compute_k_from_plan(missing_must: List[str], missing_pref: List[str], plan: dict) -> Tuple[int, int]:
    """k_must/k_pref = how many missing skills are actually targeted by the plan."""
    plan_targets = flatten_plan_targets(plan)
    target_set = {_norm_skill(x) for x in plan_targets}

    k_must = sum(1 for s in (missing_must or []) if _norm_skill(s) in target_set)
    k_pref = sum(1 for s in (missing_pref or []) if _norm_skill(s) in target_set)
    return k_must, k_pref


def what_if_after_plan(
    matched_must: int, total_must: int,
    matched_pref: int, total_pref: int,
    k_must: int, k_pref: int,
    cap: float = 95.0
) -> float:
    """Recompute readiness after plan completion using coverage math (never >100)."""
    new_matched_must = min(total_must, matched_must + k_must)
    new_matched_pref = min(total_pref, matched_pref + k_pref)

    must_pct = (new_matched_must / total_must) * 100 if total_must else 0.0
    pref_pct = (new_matched_pref / total_pref) * 100 if total_pref else 0.0

    overall = 0.75 * must_pct + 0.25 * pref_pct
    return round(min(overall, cap), 1)


def step_hours_range(step: dict) -> Tuple[float, float]:
    """
    Backward compatible helper:
    - New schema: time_hours_estimate = {"min": x, "max": y}
    - Old schema: time_hours_estimate = number
    Always returns (min_hours, max_hours)
    """
    h = step.get("time_hours_estimate", 0)

    if isinstance(h, dict):
        h_min = float(h.get("min", 0) or 0)
        h_max = float(h.get("max", h_min) or h_min)
        if h_max < h_min:
            h_min, h_max = h_max, h_min
        return h_min, h_max

    try:
        val = float(h or 0)
        return val, val
    except (TypeError, ValueError):
        return 0.0, 0.0


def format_hours_range(h_min: float, h_max: float) -> str:
    if abs(h_max - h_min) < 1e-9:
        return f"{h_min:.0f} hrs" if float(h_min).is_integer() else f"{h_min:.1f} hrs"
    return f"{h_min:.0f}–{h_max:.0f} hrs"


def is_cert_or_exam(step: dict) -> bool:
    proof = (step.get("proof_of_skill") or "").lower()
    return any(k in proof for k in ["exam", "cert", "certification", "certificate", "badge"])


def governed_step_cost(step: dict, max_step_cost: float = 500.0) -> float:
    """
    Enforce:
    - cost_estimate_usd only allowed if proof implies exam/cert
    - otherwise cost forced to 0
    """
    try:
        c = float(step.get("cost_estimate_usd", 0) or 0)
    except Exception:
        c = 0.0

    if c <= 0:
        return 0.0

    if not is_cert_or_exam(step):
        return 0.0

    return min(c, max_step_cost)


# =============================================================================
# NEW INTERACTIVE COMPONENTS
# =============================================================================

def render_semantic_audit_modal(overlap: dict, skill_name: str = None):
    """
    Display semantic audit trail in an expander.
    Shows LLM reasoning for skill matching.
    """
    semantic_audit = overlap.get("semantic_audit")
    
    if not semantic_audit:
        st.info("💡 Semantic audit not available. Using baseline keyword matching.")
        return
    
    must_items = semantic_audit.get("must", []) or []
    pref_items = semantic_audit.get("preferred", []) or []
    min_conf = semantic_audit.get("min_confidence", 0.55)
    
    st.markdown(f"**🔍 Semantic Analysis Details** (Confidence threshold: {min_conf})")
    
    # Filter by skill if specified
    if skill_name:
        must_items = [item for item in must_items if _norm_skill(item.get("skill", "")) == _norm_skill(skill_name)]
        pref_items = [item for item in pref_items if _norm_skill(item.get("skill", "")) == _norm_skill(skill_name)]
    
    if must_items:
        st.markdown("**Must-Have Skills Analysis:**")
        for item in must_items:
            confidence = float(item.get("confidence", 0))
            matched = item.get("matched", False)
            
            conf_class = "confidence-high" if confidence >= 0.75 else ("confidence-medium" if confidence >= 0.55 else "confidence-low")
            match_badge = "✅ Matched" if matched else "❌ Not Matched"
            
            st.markdown(
                f"""
                <div class="semantic-item {conf_class}">
                    <strong>{item.get('skill', 'Unknown')}</strong> 
                    <span class="badge {'badge-success' if matched else 'badge-warning'}">{match_badge}</span>
                    <span class="badge badge-info">Confidence: {confidence:.0%}</span>
                    <div class="evidence-box">
                        <strong>Evidence:</strong> {item.get('evidence', 'No evidence')}
                    </div>
                    <div class="muted">
                        <strong>Reasoning:</strong> {item.get('why', 'N/A')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    if pref_items:
        st.markdown("**Preferred Skills Analysis:**")
        for item in pref_items[:5]:  # Show top 5
            confidence = float(item.get("confidence", 0))
            matched = item.get("matched", False)
            
            conf_class = "confidence-high" if confidence >= 0.75 else ("confidence-medium" if confidence >= 0.55 else "confidence-low")
            match_badge = "✅ Matched" if matched else "❌ Not Matched"
            
            st.markdown(
                f"""
                <div class="semantic-item {conf_class}">
                    <strong>{item.get('skill', 'Unknown')}</strong> 
                    <span class="badge {'badge-success' if matched else 'badge-warning'}">{match_badge}</span>
                    <span class="badge badge-info">Confidence: {confidence:.0%}</span>
                    <div class="muted" style="margin-top: 8px;">
                        <strong>Reasoning:</strong> {item.get('why', 'N/A')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_skill_breakdown_chart(matched_must: int, missing_must: int, matched_pref: int, missing_pref: int):
    """
    Interactive stacked bar chart showing skill breakdown.
    """
    fig = go.Figure()
    
    # Must-have skills
    fig.add_trace(go.Bar(
        name='Matched Must-Have',
        x=['Skills'],
        y=[matched_must],
        marker_color=EY_GREEN,
        text=[f'{matched_must} matched'],
        textposition='inside',
    ))
    
    fig.add_trace(go.Bar(
        name='Missing Must-Have',
        x=['Skills'],
        y=[missing_must],
        marker_color=EY_ORANGE,
        text=[f'{missing_must} missing'],
        textposition='inside',
    ))
    
    # Preferred skills
    fig.add_trace(go.Bar(
        name='Matched Preferred',
        x=['Skills'],
        y=[matched_pref],
        marker_color=EY_BLUE,
        text=[f'{matched_pref} matched'],
        textposition='inside',
    ))
    
    fig.add_trace(go.Bar(
        name='Missing Preferred',
        x=['Skills'],
        y=[missing_pref],
        marker_color=EY_DARK_GRAY,
        text=[f'{missing_pref} missing'],
        textposition='inside',
    ))
    
    fig.update_layout(
        barmode='stack',
        title='Skill Match Breakdown',
        xaxis_title='',
        yaxis_title='Number of Skills',
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def render_interactive_plan_step(step: dict, step_num: int, is_expanded: bool = False):
    """
    Render a collapsible plan step with checkboxes and progress tracking.
    """
    hmin, hmax = step_hours_range(step)
    hrs = format_hours_range(hmin, hmax)
    
    resources = step.get("learning_resources", []) or []
    activities = step.get("activities", []) or []
    
    step_cost = int(governed_step_cost(step, max_step_cost=500.0))
    
    # Create unique key for this step
    step_key = f"step_{step_num}_expanded"
    
    if step_key not in st.session_state:
        st.session_state[step_key] = is_expanded
    
    # Collapsible header
    col1, col2 = st.columns([6, 1])
    
    with col1:
        if st.button(
            f"{'▼' if st.session_state[step_key] else '▶'} Week {step.get('week_range', '')} — {step.get('goal', '')}",
            key=f"toggle_{step_num}",
            use_container_width=True
        ):
            st.session_state[step_key] = not st.session_state[step_key]
    
    with col2:
        completed = st.checkbox("✓", key=f"complete_{step_num}", help="Mark as complete")
    
    # Expanded content
    if st.session_state[step_key]:
        st.markdown(
            f"""
            <div class="card" style="margin-left: 20px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                    <span><strong>⏱ Effort:</strong> {hrs}</span>
                    <span><strong>💰 Cert Cost:</strong> ${step_cost}</span>
                </div>
                
                <div style="margin-bottom: 12px;">
                    <strong>🎯 Target Skills:</strong><br>
                    {"".join([f"<span class='pill'>{s}</span>" for s in step.get('targets_skills', [])])}
                </div>
                
                <div style="margin-bottom: 12px;">
                    <strong>✅ Proof of Skill:</strong><br>
                    {step.get('proof_of_skill', 'N/A')}
                </div>
                
                <div style="margin-bottom: 12px;">
                    <strong>📋 Activities:</strong>
                    <ul style="margin-top: 8px;">
                        {"".join([f"<li>{a}</li>" for a in activities])}
                    </ul>
                </div>
                
                <div>
                    <strong>📚 Free Resources:</strong>
                    <ul style="margin-top: 8px;">
                        {"".join([f"<li>{r}</li>" for r in resources[:5]])}
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if completed:
            st.success(f"✅ Step {step_num} marked complete!")


# =============================================================================
# HR VIEW (Enhanced with interactive elements)
# =============================================================================
def render_hr_view():
    st.title("🏢 HR Talent Mobility Command Center")
    
    st.markdown(
        """
        <div class="card">
            <div class="title">📊 Upload Resumes & JDs</div>
            <div class="muted">Drag and drop files here. Max 10 MB each. DOCX, PDF supported.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.expander("📤 Upload Resumes & Job Descriptions", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            resume_files = st.file_uploader(
                "Employee Resumes (R1.docx, R2.docx, etc.)",
                type=["docx", "pdf"],
                accept_multiple_files=True,
                key="hr_resumes"
            )
        
        with col2:
            jd_files = st.file_uploader(
                "Job Descriptions (JD1.docx, JD2.docx, etc.)",
                type=["docx", "pdf"],
                accept_multiple_files=True,
                key="hr_jds"
            )
    
    # Enhanced options
    st.markdown("### ⚙️ Analysis Options")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        enable_redaction = st.checkbox("🔒 Enable PII Redaction", value=True, help="Redact emails, phones, SSNs before LLM processing")
    
    with col2:
        enable_bias_scan = st.checkbox("⚖️ Enable Bias Scan", value=True, help="Detect biased language in JDs")
    
    with col3:
        top_n = st.number_input("Top candidates per role", min_value=1, max_value=10, value=3, help="How many top candidates to show per role")
    
    with col4:
        score_mode = st.selectbox("Scoring Mode", ["hybrid", "baseline", "semantic"], index=0, help="Hybrid combines keyword + LLM semantic matching")
    
    # Run analysis button with loading state
    if st.button("🚀 Run HR Analysis", type="primary", use_container_width=True):
        if not resume_files or not jd_files:
            st.error("❌ Please upload both resumes and job descriptions.")
        else:
            with st.spinner("🔄 Analyzing talent pool... This may take 1-2 minutes..."):
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Save files
                status_text.text("📁 Saving uploaded files...")
                progress_bar.progress(10)
                
                data_dir = Path("data/hr_upload")
                data_dir.mkdir(parents=True, exist_ok=True)
                
                employees = []
                for i, rf in enumerate(resume_files):
                    path = save_upload(rf, data_dir)
                    resume_text = load_text_any(str(path))
                    emp_name = infer_name_from_resume_text(resume_text) or f"Employee {i+1}"
                    employees.append({
                        "employee_id": f"E{i+1:03d}",
                        "name": emp_name,
                        "resume_path": str(path),
                    })
                
                progress_bar.progress(30)
                
                roles = []
                for j, jf in enumerate(jd_files):
                    path = save_upload(jf, data_dir)
                    roles.append({
                        "role_id": f"R{j+1:03d}",
                        "role_title": Path(jf.name).stem.replace("_", " "),
                        "role_label": Path(jf.name).stem,
                        "jd_path": str(path),
                    })
                
                progress_bar.progress(50)
                
                # Run batch processing
                status_text.text("🤖 Processing with AI (LLM extraction, scoring, plan generation)...")
                
                run_dir = Path("outputs/hr_run") / datetime.now().strftime("%Y%m%d_%H%M%S")
                
                results = process_batch(
                    employees=employees,
                    roles=roles,
                    run_dir=str(run_dir),
                    enable_redaction=enable_redaction,
                    enable_bias_scan=enable_bias_scan,
                    hr_top_n=int(top_n),
                    deep_dive_top_n=0,  # HR view doesn't need deep dive
                    score_mode=score_mode,
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                st.session_state["hr_results"] = results
                st.success(f"✅ Analyzed {len(employees)} employees against {len(roles)} roles!")
    
    # Display results if available
    if "hr_results" in st.session_state:
        results = st.session_state["hr_results"]
        hr_role_view = results.get("hr_role_view", {})
        
        if not hr_role_view:
            st.warning("No HR role view returned.")
            return
        
        st.divider()
        st.markdown("## 🎯 Role-wise Top Talent & Upskilling Impact")
        
        # Role selector
        role_ids = list(hr_role_view.keys())
        
        if not role_ids:
            st.info("No roles to display.")
            return
        
        selected_role_id = st.selectbox(
            "Select Role to View Candidates",
            role_ids,
            format_func=lambda rid: hr_role_view[rid].get("role_title", rid),
            key="hr_role_selector"
        )
        
        role_data = hr_role_view[selected_role_id]
        
        # Role header with bias indicator
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### 📌 {role_data.get('role_title', 'Role')}")
        
        with col2:
            bias_sev = role_data.get("bias_severity", "none")
            if bias_sev == "none":
                st.success("✅ No bias detected")
            elif bias_sev == "low":
                st.warning("⚠️ Low bias detected")
            elif bias_sev == "medium":
                st.warning("⚠️ Medium bias detected")
            else:
                st.error("❌ High bias detected")
        
        # Candidates table
        candidates = role_data.get("top_candidates", [])
        
        if not candidates:
            st.info("No candidates for this role.")
        else:
            st.markdown(f"**Top {len(candidates)} Candidates:**")
            
            for idx, cand in enumerate(candidates, 1):
                with st.expander(f"#{idx} — {cand.get('employee_name', 'Unknown')} — {cand.get('overall_pct', 0):.1f}% match"):
                    c1, c2, c3, c4 = st.columns(4)
                    
                    with c1:
                        st.metric("Overall Match", f"{cand.get('overall_pct', 0):.1f}%")
                    
                    with c2:
                        st.metric("Must-Have", f"{cand.get('must_have_pct', 0):.1f}%")
                    
                    with c3:
                        st.metric("Preferred", f"{cand.get('preferred_pct', 0):.1f}%")
                    
                    with c4:
                        missing = len(cand.get("missing_must_haves", []))
                        st.metric("Missing Must-Haves", missing, delta=f"-{missing}" if missing > 0 else None)
                    
                    # Upskilling summary
                    upskill = cand.get("upskill_summary", {})
                    
                    if upskill:
                        st.markdown("**📈 Upskilling Summary:**")
                        
                        uc1, uc2, uc3 = st.columns(3)
                        
                        with uc1:
                            skills = upskill.get("top_upskill_skills", [])
                            if skills:
                                st.markdown("**Skills to develop:**")
                                st.markdown("".join([f"<span class='pill'>{s}</span>" for s in skills[:5]]), unsafe_allow_html=True)
                        
                        with uc2:
                            tt = upskill.get("total_time_hours", {})
                            if isinstance(tt, dict):
                                st.write(f"⏱ **Time:** {tt.get('min', 0)}–{tt.get('max', 0)} hrs")
                            else:
                                st.write(f"⏱ **Time:** {tt} hrs")
                        
                        with uc3:
                            st.write(f"💰 **Cost:** ${upskill.get('total_cost_usd', 0)}")
                    
                    # Missing must-haves
                    missing_list = cand.get("missing_must_haves", [])
                    if missing_list:
                        st.markdown("**⚠️ Missing Must-Have Skills:**")
                        for skill in missing_list[:8]:
                            st.markdown(f"- {skill}")


# =============================================================================
# EMPLOYEE VIEW (Enhanced with interactive modals)
# =============================================================================
def render_employee_view():
    st.title("👤 Employee Career Portal")
    
    # Login/Logout in sidebar
    if "emp_logged_in" not in st.session_state:
        st.session_state["emp_logged_in"] = False
    
    if not st.session_state["emp_logged_in"]:
        st.markdown(
            """
            <div class="card welcome-card">
                <div class="title">👋 Welcome, Asha Patel</div>
                <div class="muted">Upload your documents to explore career opportunities and personalized upskilling plans.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        with st.form("employee_login"):
            st.markdown("### 📄 Upload Your Documents")
            
            emp_id = st.text_input("Employee ID", value="E001", help="Your employee ID")
            
            col1, col2 = st.columns(2)
            
            with col1:
                resume_upload = st.file_uploader("Your Resume", type=["docx", "pdf"], key="emp_resume")
            
            with col2:
                current_jd_upload = st.file_uploader("Current Role JD (optional)", type=["docx", "pdf"], key="emp_current_jd")
            
            target_jd_upload = st.file_uploader("Target/Open Role JD", type=["docx", "pdf"], key="emp_target_jd")
            
            submitted = st.form_submit_button("🚀 Analyze My Career Path", type="primary", use_container_width=True)
            
            if submitted:
                if not resume_upload or not target_jd_upload:
                    st.error("❌ Please upload your resume and target role JD.")
                else:
                    with st.spinner("📁 Uploading documents..."):
                        data_dir = Path("data/employee") / emp_id
                        data_dir.mkdir(parents=True, exist_ok=True)
                        
                        st.session_state["emp_id_input"] = emp_id
                        st.session_state["emp_resume_path"] = str(save_upload(resume_upload, data_dir))
                        
                        if current_jd_upload:
                            st.session_state["emp_current_jd_path"] = str(save_upload(current_jd_upload, data_dir))
                        else:
                            st.session_state["emp_current_jd_path"] = None
                        
                        st.session_state["emp_target_jd_path"] = str(save_upload(target_jd_upload, data_dir))
                        
                        # Infer name from resume
                        resume_text = load_text_any(st.session_state["emp_resume_path"])
                        emp_name = infer_name_from_resume_text(resume_text) or "Employee"
                        st.session_state["emp_name"] = emp_name
                        
                        st.session_state["emp_logged_in"] = True
                        st.success(f"✅ Welcome, {emp_name}! Analyzing your profile...")
                        st.rerun()
        
        return
    
    # Logged in view
    emp_id = st.session_state.get("emp_id_input", "E001")
    emp_name = st.session_state.get("emp_name", "Employee")
    
    # Logout button in sidebar
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        for k in [
            "emp_logged_in",
            "emp_name",
            "emp_resume_path",
            "emp_current_jd_path",
            "emp_target_jd_path",
            "employee_results",
            "emp_id_input",
            "emp_resume_upload",
            "emp_current_jd_upload",
            "emp_target_jd_upload",
        ]:
            st.session_state.pop(k, None)
        st.rerun()
    
    # Build role from uploaded target JD
    target_jd_path = st.session_state.get("emp_target_jd_path")
    if not target_jd_path:
        st.error("❌ Missing Target/Open Role JD. Please logout and login again.")
        return
    
    target_role_title = Path(target_jd_path).stem.replace("_", " ")
    roles = [
        {
            "role_id": "R001",
            "role_title": target_role_title,
            "role_label": Path(target_jd_path).stem,
            "jd_path": str(target_jd_path),
        }
    ]
    
    # Run analysis once
    if "employee_results" not in st.session_state:
        st.session_state["employee_results"] = None
    
    if st.session_state["employee_results"] is None:
        with st.spinner("🤖 Analyzing your profile vs target role... This may take 30-60 seconds..."):
            employee = {
                "employee_id": emp_id,
                "name": emp_name,
                "resume_path": st.session_state["emp_resume_path"],
                "current_jd_path": st.session_state["emp_current_jd_path"],
            }
            run_dir = Path("outputs/employee") / emp_id / datetime.now().strftime("%Y%m%d_%H%M%S")
            
            results = process_batch(
                employees=[employee],
                roles=roles,
                run_dir=str(run_dir),
                enable_redaction=True,
                enable_bias_scan=True,
                deep_dive_top_n=1,
                hr_top_n=0,
            )
            st.session_state["employee_results"] = results
    
    results = st.session_state["employee_results"]
    matches = results.get("matches", [])
    
    if not matches:
        st.error("No results returned.")
        return
    
    match = matches[0]
    ranked_roles = match.get("ranked_roles", [])
    
    if not ranked_roles:
        st.error("No role scoring returned.")
        return
    
    r0 = ranked_roles[0]
    pct = (r0.get("overlap") or {}).get("percent_match") or {}
    overall = float(pct.get("overall_pct") or (r0.get("score", {}).get("overall", 0) * 100))
    
    # Header with match score
    st.markdown(
        f"""
        <div class="card welcome-card">
            <div class="title">🎯 Your Career Match: {overall:.1f}%</div>
            <div class="muted">Role: {role_display_name(r0.get('role_title','Role'), r0.get('role_label',''), r0.get('role_id',''))}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    deep_dives = match.get("deep_dive", []) or []
    
    if not deep_dives:
        st.warning("Deep dive not returned. Ensure engine deep_dive_top_n=1.")
        return
    
    d = deep_dives[0]
    
    # === INTERACTIVE FEATURES ===
    
    # Tab interface for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎓 Upskilling Plan", "👔 Manager Insights", "🔍 Semantic Analysis"])
    
    with tab1:
        # Overview tab
        score = (r0.get("score") or {})
        overlap = (r0.get("overlap") or {})
        
        km = (overlap.get("keyword_match") or {})
        kmm = (overlap.get("keyword_mismatch") or {})
        
        matched_must_list = km.get("matched_must_haves") or score.get("matched_must_haves") or []
        missing_must_list = kmm.get("missing_must_haves") or score.get("missing_must_haves") or []
        matched_pref_list = km.get("matched_preferred") or score.get("matched_preferred") or []
        missing_pref_list = kmm.get("missing_preferred") or []
        
        matched_must = len(matched_must_list)
        missing_must = missing_must_list
        total_must = matched_must + len(missing_must)
        
        matched_pref = len(matched_pref_list)
        missing_pref = missing_pref_list
        total_pref = matched_pref + len(missing_pref)
        
        must_pct = (matched_must / total_must) * 100 if total_must else 0.0
        pref_pct = (matched_pref / total_pref) * 100 if total_pref else 0.0
        
        # Skill breakdown visualization
        st.markdown("### 📊 Skill Match Breakdown")
        
        fig = render_skill_breakdown_chart(matched_must, len(missing_must), matched_pref, len(missing_pref))
        st.plotly_chart(fig, use_container_width=True)
        
        # What-If Simulator
        st.divider()
        st.markdown("### 🔮 What-If Upskilling Impact")
        
        plan_obj = d.get("plan") or {}
        k_must, k_pref = compute_k_from_plan(missing_must, missing_pref, plan_obj)
        
        current_pct = float(pct.get("overall_pct") or overall or 0.0)
        projected_pct = what_if_after_plan(
            matched_must, total_must,
            matched_pref, total_pref,
            k_must, k_pref,
            cap=95.0
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Readiness", f"{current_pct:.1f}%")
            st.progress(current_pct / 100)
        
        with col2:
            st.metric("After Plan Completion", f"{projected_pct:.1f}%", delta=f"+{(projected_pct - current_pct):.1f}%")
            st.progress(projected_pct / 100)
        
        with col3:
            improvement = projected_pct - current_pct
            if improvement > 15:
                st.success(f"🚀 High impact plan! +{improvement:.1f}%")
            elif improvement > 5:
                st.info(f"📈 Solid improvement: +{improvement:.1f}%")
            else:
                st.info(f"✅ You're nearly ready! +{improvement:.1f}%")
        
        st.caption(
            "💡 This simulation assumes successful completion of targeted skills in your upskilling plan. "
            "Readiness uses 75/25 must-have/preferred weighting, capped at 95%."
        )
        
        # Matched vs Missing skills
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Your Matched Skills")
            
            if matched_must_list:
                st.markdown("**Must-Have Skills:**")
                for skill in matched_must_list[:10]:
                    st.markdown(f"<span class='badge badge-success'>✓ {skill}</span>", unsafe_allow_html=True)
            
            if matched_pref_list:
                st.markdown("**Preferred Skills:**")
                for skill in matched_pref_list[:10]:
                    st.markdown(f"<span class='badge badge-info'>✓ {skill}</span>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("### ⚠️ Skills to Develop")
            
            if missing_must:
                st.markdown("**Missing Must-Have Skills:**")
                for skill in missing_must[:10]:
                    st.markdown(f"<span class='badge badge-warning'>⚠ {skill}</span>", unsafe_allow_html=True)
            
            if missing_pref:
                st.markdown("**Missing Preferred Skills:**")
                for skill in missing_pref[:10]:
                    st.markdown(f"<span class='badge' style='background: {EY_DARK_GRAY}; color: white;'>○ {skill}</span>", unsafe_allow_html=True)
    
    with tab2:
        # Upskilling Plan tab
        st.markdown("### 🎓 Your Personalized Upskilling Plan")
        
        title = role_display_name(d.get("role_title", "Role"), d.get("role_label", ""), d.get("role_id", ""))
        st.markdown(f"**Target Role:** {title}")
        
        # Plan summary
        summ = d.get("plan_summary", {}) or {}
        c1, c2, c3 = st.columns(3)
        
        with c1:
            tt = summ.get("total_time_hours", {"min": 0, "max": 0}) or {"min": 0, "max": 0}
            if isinstance(tt, dict):
                st.metric("⏱ Total Effort", f"{tt.get('min',0)}–{tt.get('max',0)} hrs")
            else:
                st.metric("⏱ Total Effort", f"{int(tt)} hrs")
        
        with c2:
            st.metric("📆 Estimated Weeks", int(summ.get("estimated_weeks", 0) or 0))
        
        with c3:
            st.metric("💰 Certification Cost", f"${int(summ.get('total_cost_usd', 0) or 0)}")
        
        st.markdown("**🔑 Priority Skills to Develop:**")
        skills = summ.get("top_upskill_skills", []) or []
        if skills:
            st.markdown("".join([f"<span class='pill'>{s}</span>" for s in skills]), unsafe_allow_html=True)
        else:
            st.caption("No skills returned.")
        
        # Interactive collapsible plan steps
        st.divider()
        st.markdown("### 📅 Step-by-Step Learning Path")
        
        plan = d.get("plan", {}) or {}
        
        for idx, step in enumerate((plan.get("plan", []) or []), 1):
            render_interactive_plan_step(step, idx, is_expanded=(idx == 1))
        
        # Employee explanation
        st.divider()
        st.markdown("### 💡 Why This Plan Works for You")
        st.info(d.get("employee_explanation", "No explanation available."))
    
    with tab3:
        # Manager Insights tab
        st.markdown("### 👔 Manager's Perspective on Your Growth")
        
        manager_narrative = d.get("manager_narrative", "")
        
        if manager_narrative:
            # Parse and display in structured format
            st.markdown(manager_narrative)
        else:
            st.info("No manager narrative available.")
    
    with tab4:
        # Semantic Analysis tab
        st.markdown("### 🔍 AI Reasoning Behind Your Match")
        
        st.info(
            "💡 This shows how our AI evaluated your skills using semantic understanding, "
            "not just keyword matching. Each skill shows confidence level and evidence from your resume."
        )
        
        # Semantic audit viewer
        render_semantic_audit_modal(overlap)
        
        # Option to view specific skill
        st.divider()
        st.markdown("#### 🔎 Inspect Specific Skill")
        
        all_skills = list(set(matched_must_list + [str(s) for s in missing_must] + matched_pref_list))
        
        if all_skills:
            selected_skill = st.selectbox("Select a skill to see detailed analysis:", all_skills)
            
            if selected_skill:
                with st.expander(f"Analysis for '{selected_skill}'", expanded=True):
                    render_semantic_audit_modal(overlap, skill_name=selected_skill)
        else:
            st.caption("No skills available for inspection.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    st.sidebar.title("EY Talent Mobility")
    view = st.sidebar.radio("Select View", ["🏢 HR Dashboard", "👤 Employee Portal"], key="view_mode_radio")
    
    # Add info section in sidebar
    with st.sidebar.expander("ℹ️ About This Tool"):
        st.markdown("""
        **Features:**
        - 🤖 AI-powered skill matching
        - 📊 Hybrid scoring (keyword + semantic)
        - 🎓 Personalized upskilling plans
        - ⚖️ Bias detection in JDs
        - 🔒 PII redaction for privacy
        - 🔍 Explainable AI with semantic audit
        
        **Built for SCLC 2026 EY Challenge**
        """)
    
    if "HR" in view:
        render_hr_view()
    else:
        render_employee_view()


if __name__ == "__main__":
    main()