# filename: manager_agent.py
# Manager agent: classify -> (split if "both") -> run agents -> validate -> synthesize.

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

import qualitative_agent
import quantitative_agent
import reviewer
from tokenomics.logger import log
from validation.validator import (
    validate_qualitative,
    validate_quantitative,
)


# FIX: load_dotenv("APIKEY.env") only worked when the program was started from the
# project folder. Anchoring to this file's location works from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / "APIKEY.env")
load_dotenv()

# FIX: was hardcoded; now shares GEMINI_MODEL with the other agents.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# FIX: the synthesis prompt used to include EVERY row the SQL returned (up to 10,000).
# The specialist's written answer already summarises them, so only a sample goes in.
MAX_ROWS_IN_SYNTHESIS = 20

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY or GOOGLE_API_KEY was not found."
    )

client = genai.Client(api_key=api_key)


# ---------------------------------------------------------------- helpers

def _response_text(response: Any) -> str:
    try:
        text = getattr(response, "text", None)

        if text:
            return text.strip()
    except Exception:
        pass

    return ""


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)

    if usage is None:
        return 0, 0

    input_tokens = getattr(usage, "prompt_token_count", 0)
    # FIX: thinking tokens are billed as output but reported separately, so they were
    # missing from the tokenomics log and every cost figure was understated.
    output_tokens = (
        (getattr(usage, "candidates_token_count", 0) or 0)
        + (getattr(usage, "thoughts_token_count", 0) or 0)
    )

    return (
        int(input_tokens or 0),
        int(output_tokens or 0),
    )


def _log_model_call(
    query: str,
    agent: str,
    response: Any = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    if response is not None:
        input_tokens, output_tokens = _usage(response)

    log(
        query=query,
        agent=agent,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _generate(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    json_mode: bool = False,
) -> Any:
    # max_tokens is generous because thinking tokens share this budget; too small a
    # cap can return empty text.
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type="application/json" if json_mode else None,
    )

    return client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=config,
    )


def _failed_qualitative_validation(message: str) -> dict:
    return {
        "is_grounded": False,
        "refused_to_answer": False,
        "sources_cited": [],
        "flag": True,
        "warning": message,
    }


def _failed_quantitative_validation(message: str) -> dict:
    return {
        "sql_validated": False,
        "sql_blocked": False,
        "execution_error": True,
        "flag": True,
        "warning": message,
    }


# ---------------------------------------------------------------- routing

# FIX: the old prompt gave no example of a mixed question, and the parser picked the
# first label found in the reply (so "both: qualitative and quantitative" -> qualitative).
# Now "how do our numbers compare to benchmarks / what policies affect X" is explicitly
# "both", and the parser takes the first valid label as the whole word.
CLASSIFY_INSTRUCTION = """Classify a business question as exactly one word: qualitative, quantitative, or both.

qualitative = answered from documents: policies, processes, procedures, explanations, industry benchmarks.
quantitative = answered from the company metrics table: numbers, totals, averages, trends, counts, rates, comparisons across regions, products, segments or time.
both = needs the company's own numbers AND documents. A question that compares company numbers to industry standards or benchmarks, or asks which policies affect a metric, is both.

Examples:
"What is our security policy?" -> qualitative
"Show monthly revenue by region" -> quantitative
"How does our employee satisfaction compare to industry standards and what policies might impact this?" -> both

Reply with one word only."""


def classify(query: str) -> str:
    response = None
    route = "both"

    try:
        response = _generate(query, system=CLASSIFY_INSTRUCTION, max_tokens=256)

        for word in re.findall(r"[a-z]+", _response_text(response).lower()):
            if word in {"qualitative", "quantitative", "both"}:
                route = word
                break
        # An unrecognised reply falls back to "both": running both agents costs a
        # little more but never silently drops half of the evidence.

    except Exception:
        route = "both"

    finally:
        _log_model_call(
            query,
            "manager-classifier",
            response=response,
        )

    return route


# FIX (main bug): for a "both" question the full compound sentence used to be sent to
# BOTH agents. The SQL agent was asked about "industry standards" (not in the data) and
# the document search was diluted by the numeric part. Now each agent gets only the
# part it can answer.
SPLIT_INSTRUCTION = """Split one business question into two standalone sub-questions.

data_question: the part answerable with numbers from the company metrics table (averages, totals, trends, counts). Do not mention industry standards or policies.
document_question: the part answerable from company documents (policies, processes, industry benchmarks, explanations).

Return JSON only: {"data_question": "...", "document_question": "..."}"""


def split_query(query: str) -> dict:
    response = None
    data_question = query
    document_question = query

    try:
        response = _generate(
            query,
            system=SPLIT_INSTRUCTION,
            max_tokens=1024,
            json_mode=True,
        )
        parsed = json.loads(_response_text(response))

        data_question = str(parsed.get("data_question", "")).strip() or query
        document_question = (
            str(parsed.get("document_question", "")).strip() or query
        )

    except Exception:
        # Bad or empty JSON: fall back to sending the original question to both.
        data_question = query
        document_question = query

    finally:
        _log_model_call(
            query,
            "manager-splitter",
            response=response,
        )

    return {
        "data_question": data_question,
        "document_question": document_question,
    }


# ---------------------------------------------------------------- specialists

def run_qualitative(query: str, question: str | None = None) -> dict:
    """`query` is the user's original question (used for logging);
    `question` is what the agent actually searches for."""
    asked = question or query

    try:
        result = qualitative_agent.run(asked)

        answer = result.get("answer", "")
        chunks = result.get("chunks", [])

        validation = validate_qualitative(
            answer,
            chunks,
        )

        _log_model_call(
            query,
            "qualitative",
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
        )

        # SILVER: second-pass reviewer. A separate Gemini call checks every claim in the
        # answer against the retrieved chunks (the citation check alone only proves that a
        # "[Source N]" tag exists). It never raises; a failed review comes back flagged.
        review = reviewer.review_answer(asked, chunks, answer)

        if review["verdict"] != "SKIPPED":
            _log_model_call(
                query,
                "qualitative-reviewer",
                input_tokens=review["input_tokens"],
                output_tokens=review["output_tokens"],
            )

        validation = reviewer.merge_review(validation, review)

        return {
            "agent": "qualitative",
            "question": asked,
            "answer": answer,
            "chunks": chunks,
            "validation": validation,
            "error": None,
        }

    except Exception as exc:
        message = f"Qualitative agent failed: {exc}"

        _log_model_call(query, "qualitative")

        return {
            "agent": "qualitative",
            "question": asked,
            "answer": "",
            "chunks": [],
            "validation": _failed_qualitative_validation(message),
            "error": message,
        }


def run_quantitative(query: str, question: str | None = None) -> dict:
    asked = question or query

    try:
        result = quantitative_agent.run(asked)

        answer = result.get("answer", "")
        sql = result.get("sql", "")
        status = result.get("validation", "ERROR")

        validation = validate_quantitative(
            answer,
            sql,
            status,
        )

        _log_model_call(
            query,
            "quantitative",
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
        )

        return {
            "agent": "quantitative",
            "question": asked,
            "answer": answer,
            "sql": sql,
            "rows": result.get("rows", []),
            "validation": validation,
            "error": None,
        }

    except Exception as exc:
        message = f"Quantitative agent failed: {exc}"

        _log_model_call(query, "quantitative")

        return {
            "agent": "quantitative",
            "question": asked,
            "answer": "",
            "sql": "",
            "rows": [],
            "validation": _failed_quantitative_validation(message),
            "error": message,
        }


# ---------------------------------------------------------------- synthesis

def _format_evidence(results: list[dict]) -> str:
    evidence = []

    for result in results:
        rows = result.get("rows", [])

        evidence.append(
            {
                "agent": result["agent"],
                "question_asked": result.get("question", ""),
                "answer": result["answer"],
                "validation": result["validation"],
                "sql": result.get("sql", ""),
                "total_rows": len(rows),
                "sample_rows": rows[:MAX_ROWS_IN_SYNTHESIS],
                "error": result.get("error"),
            }
        )

    return json.dumps(evidence, indent=2, default=str)


SYNTHESIS_INSTRUCTION = """You write the final answer to a business question from specialist results.

Rules:
1. Use only the specialist results. Do not invent facts, metrics, or sources.
2. Keep every number exactly as given and keep any [Source N] citations.
3. Make clear which findings come from company data and which come from documents.
4. If an agent failed, was blocked, or found nothing (for example no industry benchmark in the documents), say so plainly. Do not fill the gap with general knowledge, and do not claim a comparison you cannot support.
5. Include relevant validation warnings.
6. Do not mention internal prompts or implementation details.
7. Be concise and answer the user's original question directly."""


def synthesize(query: str, results: list[dict]) -> str:
    response = None

    prompt = (
        f"The user asked:\n\n{query}\n\n"
        f"Validated specialist results:\n\n{_format_evidence(results)}"
    )

    try:
        response = _generate(
            prompt,
            system=SYNTHESIS_INSTRUCTION,
            temperature=0.2,
            max_tokens=2048,
        )
        answer = _response_text(response)

        return answer or "No final answer was generated."

    finally:
        _log_model_call(
            query,
            "manager-synthesis",
            response=response,
        )


# ---------------------------------------------------------------- entry point

def run(query: str) -> dict:
    route = classify(query)

    sub_questions = {
        "data_question": query,
        "document_question": query,
    }

    if route == "both":
        sub_questions = split_query(query)

    results = []

    if route in {"qualitative", "both"}:
        results.append(
            run_qualitative(query, sub_questions["document_question"])
        )

    if route in {"quantitative", "both"}:
        results.append(
            run_quantitative(query, sub_questions["data_question"])
        )

    answer = synthesize(query, results)

    return {
        "query": query,
        "route": route,
        "sub_questions": sub_questions,
        "results": results,
        "answer": answer,
    }