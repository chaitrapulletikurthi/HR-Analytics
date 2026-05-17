# scripts/preparse_resumes.py
"""
One-time / on-demand script: Parse resumes → save structured JSON (only parsed fields + metadata)
- Tries to extract full name and employee ID from resume TEXT (not just filename)
- Saves NO matching scores — only resume extraction result + metadata
"""

import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
import datetime as dt  # for UTC
from openai import OpenAI
import re

# Try to add project root to sys.path automatically
project_root = Path(__file__).resolve().parent.parent  # scripts/ → project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.talent_agent.extract import extract_resume, DEFAULT_MODEL
    from src.talent_agent.io import load_text_any, redact_pii
except ImportError as e:
    print("Import error:", e)
    print("\nCurrent working directory:", Path.cwd())
    print("Make sure you run this script from the project root folder:")
    print("cd C:\\Users\\chait\\Documents\\SCLC\\talent_agent")
    print("python scripts\\preparse_resumes.py")
    sys.exit(1)

client = OpenAI()  # Make sure OPENAI_API_KEY is set in your environment

# Folders (relative to project root)
RAW_RESUMES_FOLDER = Path("data/raw/resumes")
EMP_DB_FOLDER = Path("data/employees_db")

RAW_RESUMES_FOLDER.mkdir(parents=True, exist_ok=True)
EMP_DB_FOLDER.mkdir(parents=True, exist_ok=True)


def extract_name_and_id_from_text(text: str) -> tuple[str | None, str | None]:
    """
    Cheap LLM call to extract full name and employee ID from resume header.
    Returns (full_name, employee_id) or (None, None)
    """
    prompt = f"""
You are extracting basic personal info from the top of a resume.
Return STRICT JSON with exactly two keys:

{{
  "full_name": "Full name of the person (first + last, e.g. Sarah Johnson)",
  "employee_id": "Any employee/staff/ID number mentioned (e.g. Emp ID: 12345, Staff #: ABC-001), or null if none found"
}}

Rules:
- Look only at the beginning of the text (header/contact section)
- Do NOT invent values
- If unclear or missing, use null

Resume header text:
{text[:2500]}   # usually enough for name/ID
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150
        )
        content = resp.choices[0].message.content.strip()

        # Debug: show raw response if needed
        # print("Raw LLM response for name/ID:", content)

        # Robust cleaning: remove markdown fences, extra whitespace
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].strip()
        elif content.startswith("{"):
            pass  # already good

        data = json.loads(content)
        name = data.get("full_name")
        eid = data.get("employee_id")

        return (
            name.strip() if isinstance(name, str) and name.strip() and name.lower() != "null" else None,
            eid.strip() if isinstance(eid, str) and eid.strip() and eid.lower() != "null" else None
        )
    except json.JSONDecodeError as je:
        print(f"  JSON parse error in name/ID response: {je}")
        print(f"  Raw content was: {content[:300]}...")  # show beginning for debug
        return None, None
    except Exception as e:
        print(f"  Name/ID extraction failed: {e}")
        return None, None


def parse_and_save_resume(
    resume_file: Path,
    employee_id: str = None,
    full_name: str = None,
    model: str = DEFAULT_MODEL
) -> Path | None:
    """
    Parse one resume and save clean JSON (no scores).
    Now tries to extract name & ID from content first.
    """
    if not resume_file.is_file():
        print(f"File not found: {resume_file}")
        return None

    print(f"\nProcessing: {resume_file.name}")

    try:
        raw_text = load_text_any(resume_file)
        redacted_text, redaction_count = redact_pii(raw_text)

        print(f"  → Redacted {redaction_count} PII items")

        # Step 1: Try to extract name & ID from text
        extracted_name, extracted_eid = extract_name_and_id_from_text(redacted_text)

        # Step 2: Fallback to filename-based guess if LLM didn't find anything
        if not employee_id:
            stem = resume_file.stem.upper()
            match = re.match(r'^([ER])(\d+)(?:_(.+))?$', stem, re.IGNORECASE)
            if match:
                prefix, num, _ = match.groups()
                employee_id = f"{prefix.upper()}{num.zfill(3)}"  # R1 → R001, E5 → E005
            else:
                employee_id = f"EMP_{stem.replace(' ', '_').upper()}"

        # Prefer extracted values
        final_id = extracted_eid or employee_id
        final_name = extracted_name or full_name or f"Employee {final_id}"

        print(f"  → Final ID: {final_id} | Name: {final_name}")

        # Step 3: Main resume parsing
        parsed = extract_resume(client, redacted_text, model=model)

        record = {
            "employee_id": final_id,
            "full_name": final_name,
            "parsed_resume": parsed,
            "last_parsed_hash": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "parsing_date": dt.datetime.now(dt.UTC).isoformat(),
            "parsing_model": model,
            "source_file": str(resume_file.relative_to(RAW_RESUMES_FOLDER.parent)),
            "redactions_count": redaction_count
        }

        out_file = EMP_DB_FOLDER / f"{final_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"  Saved: {out_file}")
        return out_file

    except Exception as e:
        print(f"  Failed: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-parse resumes into reusable employee DB (JSON)")
    parser.add_argument("--folder", type=str, default=str(RAW_RESUMES_FOLDER),
                        help="Folder with resumes")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model to use for extraction")

    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    print(f"Scanning: {folder}")
    print(f"Using model: {args.model}\n")

    count = 0
    for pattern in ["*.docx", "*.pdf", "*.txt"]:
        for file_path in folder.glob(pattern):
            parse_and_save_resume(file_path, model=args.model)
            count += 1

    print(f"\nDone. Processed {count} file(s).")
    print(f"Parsed data saved in: {EMP_DB_FOLDER.resolve()}")