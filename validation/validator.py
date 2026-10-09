
import re


def _is_refusal(answer: str) -> bool:
    refusal_phrases = (
        "cannot find",
        "can't find",
        "unable to find",
        "not enough information",
        "no relevant information",
        "cannot answer",
        "can't answer",
    )

    normalized = answer.lower()
    return any(phrase in normalized for phrase in refusal_phrases)


def validate_qualitative(
    answer: str,
    chunks: list[dict],
) -> dict:
    answer = answer or ""
    sources_cited = []

    for index, chunk in enumerate(chunks, start=1):
        citation_markers = (
            f"Source {index}",
            f"[Source {index}]",
        )

        if any(marker in answer for marker in citation_markers):
            source = chunk.get("source")

            if source and source not in sources_cited:
                sources_cited.append(source)

    refused = _is_refusal(answer)
    grounded = len(sources_cited) > 0
    missing_answer = not answer.strip()

    if missing_answer:
        warning = "The agent returned an empty answer."
    elif not grounded and not refused:
        warning = "Response may not be grounded in source documents."
    else:
        warning = None

    return {
        "is_grounded": grounded,
        "refused_to_answer": refused,
        "sources_cited": sources_cited,
        "flag": missing_answer or (not grounded and not refused),
        "warning": warning,
    }


def validate_quantitative(
    answer: str,
    sql: str,
    validation_status: str,
) -> dict:
    status = (validation_status or "").upper()
    answer_present = bool((answer or "").strip())
    sql_present = bool((sql or "").strip())

    warnings = []

    if status != "PASSED":
        warnings.append(f"SQL validation status: {status or 'UNKNOWN'}")

    if status == "PASSED" and not sql_present:
        warnings.append("No SQL query was returned.")

    if status == "PASSED" and not answer_present:
        warnings.append("The quantitative agent returned an empty answer.")

    return {
        "sql_validated": status == "PASSED",
        "sql_blocked": status == "FAILED",
        "execution_error": status == "ERROR",
        "flag": bool(warnings),
        "warning": " ".join(warnings) if warnings else None,
    }
