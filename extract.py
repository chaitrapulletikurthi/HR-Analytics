from __future__ import annotations
import json
from openai import OpenAI
import re

DEFAULT_MODEL = "gpt-4o-mini"



def _safe_json_loads(s: str) -> dict:
    """
    Robust JSON loader for LLM output.
    - strips markdown fences
    - extracts first JSON object
    - removes invalid control characters that break json.loads
    """
    s = (s or "").strip()

    # Remove markdown fences if present
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1].strip()

    # Extract first JSON object region if extra text exists
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]

    # Remove ASCII control characters except \t \n \r (even those can break inside strings)
    # This is a pragmatic sanitizer for LLM-generated JSON
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)

    # Also normalize Windows smart quotes sometimes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")

    return json.loads(s)


def _responses_text(resp) -> str:
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    # fallback (older sdk shapes)
    if hasattr(resp, "model_dump"):
        d = resp.model_dump()
        if isinstance(d, dict) and d.get("output_text"):
            return d["output_text"]
        return json.dumps(d)
    return str(resp)

def extract_resume(client: OpenAI, resume_text: str, model: str = DEFAULT_MODEL) -> dict:
    prompt = f"""
You are an HR analytics extraction engine. Extract ONLY from the provided resume text.
Return STRICT JSON with keys:
- skills: list of objects {{name, evidence}}
- titles: list of strings
- years_experience: number (best estimate)
- domains: list of strings (e.g., marketing analytics, finance, BI)

Rules:
- Include up to 25 skills/tools, each MUST include a short evidence quote from the text.
- Do not invent.

Resume text:
{resume_text}
""".strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
    )
    return _safe_json_loads(_responses_text(resp))

def extract_jd(client: OpenAI, jd_text: str, model: str = DEFAULT_MODEL) -> dict:
    prompt = f"""
You are an HR analytics job description parser.
Return STRICT JSON with keys:
- role_title: string
- must_have_skills: list of strings
- preferred_skills: list of strings
- responsibilities: list of strings (max 8)

Rules:
- Use only what is present in the JD.
- Keep skills normalized (short phrases).
- Do not invent.

JD text:
{jd_text}
""".strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
    )
    return _safe_json_loads(_responses_text(resp))
