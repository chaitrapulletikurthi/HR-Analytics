from pathlib import Path
import pandas as pd

from src.talent_agent.engine import process_batch

DATA_DIR = Path("data")
OUT_DIR = Path("outputs") / "run_demo"

def main():
    # ---- Roles (edit this list to match your filenames) ----
    roles = [
        {"role_id": "R001", "role_title": "Role 1", "jd_path": str(DATA_DIR / "JD1.docx")},
        {"role_id": "R002", "role_title": "Role 2", "jd_path": str(DATA_DIR / "JD2.docx")},
        {"role_id": "R003", "role_title": "Role 3", "jd_path": str(DATA_DIR / "JD3.docx")},
        {"role_id": "R004", "role_title": "Role 4", "jd_path": str(DATA_DIR / "JD4.docx")},
        {"role_id": "R005", "role_title": "Role 5", "jd_path": str(DATA_DIR / "JD5.docx")},
    ]

    # ---- Employees (edit this list to match your filenames) ----
    # Option A: one shared CURRENT_JD for everyone:
    shared_current = DATA_DIR / "CURRENT_JD.docx"
    use_shared_current = shared_current.exists()

    employees = []
    for i in range(1, 6):
        emp_id = f"E{i:03d}"
        resume_path = DATA_DIR / f"R{i}.docx"
        if use_shared_current:
            current_jd_path = str(shared_current)
        else:
            # Option B: per-employee current JD like CURRENT_E001.docx
            cj = DATA_DIR / f"CURRENT_{emp_id}.docx"
            current_jd_path = str(cj) if cj.exists() else None

        employees.append({
            "employee_id": emp_id,
            "name": f"Employee {i}",
            "resume_path": str(resume_path),
            "current_jd_path": current_jd_path
        })

    results = process_batch(
        employees=employees,
        roles=roles,
        run_dir=str(OUT_DIR),
        enable_redaction=True,
        enable_bias_scan=True,
        model="gpt-4o-mini",
    )

    print("\n=== BATCH COMPLETE ===")
    print("Errors:", len(results["errors"]))
    print("Artifacts:", results["artifact_paths"])

    df = pd.DataFrame(results["summary_table"])
    print("\nSUMMARY TABLE")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
