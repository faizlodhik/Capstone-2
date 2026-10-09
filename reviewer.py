# reviewer.py
# Silver tier: second validation strategy for the qualitative agent.
#
# The citation check in validator.py only confirms that a "[Source N]" appears in the answer.
# It cannot tell whether Source N actually says what the answer claims. This module makes a
# SECOND Gemini call that acts as a strict reviewer: it checks every claim in the answer
# against the retrieved chunks and flags unsupported or mis-cited claims.
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / "APIKEY.env")

REVIEWER_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Generous cap: thinking tokens share this budget, and a truncated reply can't be parsed.
REVIEW_MAX_OUTPUT_TOKENS = 2048
VALID_VERDICTS = {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"}
REFUSAL_MARKER = "cannot find"

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is missing from APIKEY.env.")
        _client = genai.Client(api_key=api_key)
    return _client


# Reviewer is deliberately given a different job from the answering prompt: it must not
# answer the question, only judge whether the answer is backed by the sources.
# Temperature 0 so the same answer gets the same verdict.
REVIEW_SYSTEM_INSTRUCTION = """You are a strict fact-checking reviewer. You are given SOURCES, a QUESTION, and an ANSWER written by another assistant.

Judge ONLY whether the ANSWER is supported by the SOURCES. Use no outside knowledge. Do not judge writing quality or whether the answer is complete.

Steps:
1. Break the ANSWER into individual factual claims (numbers, deadlines, names, requirements).
2. For each claim, check the source it cites ([Source N]). The claim must be stated in, or clearly follow from, that source.
3. List claims that no source supports under unsupported_claims. This includes wrong numbers or details that differ from the source.
4. List claims that ARE stated in the sources but cite the wrong source number under miscited_claims.

Verdict:
- SUPPORTED: every claim is supported and correctly cited.
- PARTIALLY_SUPPORTED: some claims are unsupported or miscited, others are fine.
- UNSUPPORTED: the main claims are not supported by the sources.

Return JSON only, with no markdown:
{"verdict": "SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED", "unsupported_claims": ["..."], "miscited_claims": ["..."], "explanation": "one sentence"}"""


def build_review_prompt(question: str, chunks: list[dict], answer: str) -> str:
    sources = "\n\n".join(
        f"[Source {i}: {c['source']}]\n{c['content']}"
        for i, c in enumerate(chunks, start=1)
    )
    return f"SOURCES:\n{sources}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}"


def _tokens(response) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0
    inp = getattr(usage, "prompt_token_count", 0) or 0
    out = (getattr(usage, "candidates_token_count", 0) or 0) + (
        getattr(usage, "thoughts_token_count", 0) or 0
    )
    return inp, out


def parse_review(text: str) -> dict:
    """Parse the reviewer's JSON. Raises ValueError if it is unusable."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    parsed = json.loads(cleaned)

    verdict = str(parsed.get("verdict", "")).strip().upper()
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"unrecognised verdict: {verdict!r}")

    return {
        "verdict": verdict,
        "unsupported_claims": [str(c) for c in parsed.get("unsupported_claims", []) or []],
        "miscited_claims": [str(c) for c in parsed.get("miscited_claims", []) or []],
        "explanation": str(parsed.get("explanation", "")).strip(),
    }


def _result(verdict: str, flag: bool, warning, inp=0, out=0, **extra) -> dict:
    base = {
        "verdict": verdict,
        "flag": flag,
        "warning": warning,
        "unsupported_claims": [],
        "miscited_claims": [],
        "explanation": "",
        "input_tokens": inp,
        "output_tokens": out,
    }
    base.update(extra)
    return base


def review_answer(question: str, chunks: list[dict], answer: str) -> dict:
    """Second-pass review. Never raises: failures come back as verdict ERROR (flagged),
    so a broken reviewer can never silently pass an answer."""
    # Nothing to review: refusals make no claims, and with no chunks there is no evidence.
    if not answer or not chunks or REFUSAL_MARKER in answer.lower():
        return _result("SKIPPED", False, None)

    try:
        response = _get_client().models.generate_content(
            model=REVIEWER_MODEL,
            contents=build_review_prompt(question, chunks, answer),
            config=types.GenerateContentConfig(
                system_instruction=REVIEW_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.0,
                max_output_tokens=REVIEW_MAX_OUTPUT_TOKENS,
            ),
        )
    except Exception as exc:
        return _result("ERROR", True, f"Reviewer could not run: {exc}")

    inp, out = _tokens(response)

    try:
        review = parse_review(response.text)
    except Exception as exc:
        return _result("ERROR", True, f"Reviewer returned unusable output: {exc}", inp, out)

    if review["verdict"] == "SUPPORTED":
        return _result("SUPPORTED", False, None, inp, out, **{
            k: review[k] for k in ("unsupported_claims", "miscited_claims", "explanation")
        })

    details = []
    if review["unsupported_claims"]:
        details.append("unsupported: " + "; ".join(review["unsupported_claims"]))
    if review["miscited_claims"]:
        details.append("wrong citation: " + "; ".join(review["miscited_claims"]))
    warning = f"Reviewer verdict {review['verdict']}" + (
        " (" + " | ".join(details) + ")" if details else ""
    )
    return _result(review["verdict"], True, warning, inp, out, **{
        k: review[k] for k in ("unsupported_claims", "miscited_claims", "explanation")
    })


def merge_review(validation: dict, review: dict) -> dict:
    """Attach the reviewer's findings to the validator's dict (keeps its existing keys).
    The reviewer can only ADD a flag, never clear one the validator raised."""
    merged = dict(validation)
    merged["reviewer_verdict"] = review["verdict"]
    merged["reviewer_unsupported_claims"] = review["unsupported_claims"]
    merged["reviewer_miscited_claims"] = review["miscited_claims"]
    merged["reviewer_explanation"] = review["explanation"]

    if review["flag"]:
        merged["flag"] = True
        parts = [w for w in (validation.get("warning"), review["warning"]) if w]
        merged["warning"] = " | ".join(parts)

    return merged


if __name__ == "__main__":
    # Quick manual demo: a deliberately wrong answer should be flagged.
    demo_chunks = [{"source": "security_policy.txt",
                    "content": "Departing personnel must have access removed within four hours."}]
    demo_answer = "Departing personnel must have access removed within 24 hours [Source 1]."
    print(json.dumps(review_answer("How fast is access removed?", demo_chunks, demo_answer), indent=2))