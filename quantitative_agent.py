# filename: quantitative_agent.py
# Quantitative agent: natural language -> SQL (Gemini) -> validate -> run on an in-memory
# SQLite copy of business_metrics.csv -> interpret results (Gemini).

from __future__ import annotations

import csv
import os
import re
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent if HERE.name == "agents" else HERE

load_dotenv(PROJECT_ROOT / "APIKEY.env")
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)

MAX_ROWS_TO_MODEL = 20
MAX_QUERY_ROWS = 10_000
QUERY_TIMEOUT_SECONDS = 10

# FIX: these budgets were 512. On Gemini models that "think", thinking tokens are
# drawn from max_output_tokens, so a complex question could burn the whole budget
# and return EMPTY text (shown as "Gemini returned empty SQL"). Larger caps cost
# nothing unless they are actually used.
SQL_MAX_OUTPUT_TOKENS = 2048
INTERPRET_MAX_OUTPUT_TOKENS = 1024

_client = None


# FIX: rule 7 used to say "if the question cannot be answered, return UNANSWERABLE".
# A mixed question ("compare to industry standards and what policies affect it") can
# never be fully answered from a metrics table, so it was always rejected even though
# half of it IS answerable. Now: answer the part the data supports, and only return
# UNANSWERABLE when none of it is answerable.
# Rule 8 gives computed columns descriptive aliases so the definition used (e.g. what
# "churn rate" means) is visible in the output and easy to verify by hand.
SQL_SYSTEM_INSTRUCTION = """
Convert the business question into one SQLite query.

Rules:
1. Return only SQL. No markdown or explanation.
2. Use SQLite syntax only.
3. Use only the table and columns in the supplied schema.
4. Return exactly one read-only SELECT statement.
5. A WITH query is allowed if it ultimately returns data.
6. Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or PRAGMA.
7. If only PART of the question can be answered from the schema (for example it also
   asks about industry standards or policies, which are not in the data), write SQL
   for the part the data can answer and ignore the rest.
8. If a requested metric is not a column (for example a rate), compute it from the
   columns and give the result a descriptive alias that states the definition used.
9. Only if NONE of the question can be answered from the schema, return exactly:
   SELECT 'UNANSWERABLE' AS error
"""


INTERPRET_SYSTEM_INSTRUCTION = """
Explain the SQL results clearly to a business user.

Rules:
1. Use only the supplied query results.
2. Do not invent, estimate, or add external figures.
3. If no rows were returned, say no matching data was found.
4. Mention if the displayed results were truncated.
5. State important assumptions made by the SQL (for example how a rate was defined,
   or which years were included).
6. Be concise.
"""


# FIX: REPLACE removed from this list. replace() is a normal SQLite string function and
# was being rejected. Writes are still impossible: the authorizer denies INSERT/UPDATE/
# DELETE/DROP/etc. (which also covers "REPLACE INTO"), and the connection is query_only.
BLOCKED_KEYWORDS = re.compile(
    r"\b("
    r"DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|"
    r"ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|ANALYZE"
    r")\b",
    re.IGNORECASE,
)


def get_client() -> genai.Client:
    global _client

    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is missing from APIKEY.env or .env."
            )

        _client = genai.Client(api_key=api_key)

    return _client


def get_token_counts(response) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)

    if usage is None:
        return 0, 0

    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = (
        getattr(usage, "candidates_token_count", 0) or 0
    ) + (getattr(usage, "thoughts_token_count", 0) or 0)

    return input_tokens, output_tokens


def find_business_metrics_csv() -> Path:
    """Find the CSV regardless of whether it is under data/documents or documents."""

    expected_paths = [
        PROJECT_ROOT / "data" / "documents" / "business_metrics.csv",
        PROJECT_ROOT / "documents" / "business_metrics.csv",
        PROJECT_ROOT / "data" / "business_metrics.csv",
        PROJECT_ROOT / "business_metrics.csv",
    ]

    for path in expected_paths:
        if path.is_file():
            return path

    ignored_folders = {
        "venv",
        ".venv",
        "__pycache__",
        ".git",
        "site-packages",
    }

    matches = []

    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue

        if path.name.lower() != "business_metrics.csv":
            continue

        if any(part.lower() in ignored_folders for part in path.parts):
            continue

        matches.append(path)

    if not matches:
        raise FileNotFoundError(
            "business_metrics.csv was not found. "
            "Checked data/documents, documents, data, and the project root."
        )

    if len(matches) > 1:
        locations = "\n".join(str(path) for path in matches)
        raise RuntimeError(
            f"Multiple business_metrics.csv files were found:\n{locations}"
        )

    return matches[0]


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def normalize_name(value: str, fallback: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    name = name.strip("_") or fallback

    if name[0].isdigit():
        name = f"column_{name}"

    return name


def make_unique_headers(headers: list[str]) -> list[str]:
    result = []
    used = set()

    for index, header in enumerate(headers, start=1):
        base = normalize_name(header, f"column_{index}")
        candidate = base
        suffix = 2

        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1

        used.add(candidate)
        result.append(candidate)

    return result


def infer_type(values: list[str]) -> str:
    nonempty = [value.strip() for value in values if value.strip()]

    if not nonempty:
        return "TEXT"

    if all(re.fullmatch(r"-?\d+", value) for value in nonempty):
        return "INTEGER"

    try:
        for value in nonempty:
            float(value)
        return "REAL"
    except ValueError:
        return "TEXT"


def convert_value(value: str, data_type: str):
    value = value.strip()

    if not value:
        return None

    if data_type == "INTEGER":
        return int(value)

    if data_type == "REAL":
        return float(value)

    return value


def load_csv_database() -> tuple[sqlite3.Connection, dict]:
    csv_path = find_business_metrics_csv()

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        sample = csv_file.read(4096)
        csv_file.seek(0)

        try:
            delimiter = csv.Sniffer().sniff(
                sample,
                delimiters=",;\t|",
            ).delimiter
        except csv.Error:
            delimiter = ","

        reader = csv.reader(csv_file, delimiter=delimiter)

        try:
            raw_headers = next(reader)
        except StopIteration as exc:
            raise ValueError(
                "business_metrics.csv is empty."
            ) from exc

        if not raw_headers:
            raise ValueError(
                "business_metrics.csv has no header row."
            )

        headers = make_unique_headers(raw_headers)

        raw_rows = [
            row for row in reader
            if any(cell.strip() for cell in row)
        ]

    rows = []

    for raw_row in raw_rows:
        row = raw_row[:len(headers)]
        row += [""] * (len(headers) - len(row))
        rows.append(row)

    data_types = [
        infer_type([row[index] for row in rows])
        for index in range(len(headers))
    ]

    connection = sqlite3.connect(":memory:")
    table_name = "business_metrics"

    column_definitions = ", ".join(
        f"{quote_identifier(column)} {data_type}"
        for column, data_type in zip(headers, data_types)
    )

    connection.execute(
        f"CREATE TABLE {quote_identifier(table_name)} "
        f"({column_definitions})"
    )

    placeholders = ", ".join("?" for _ in headers)
    insert_sql = (
        f"INSERT INTO {quote_identifier(table_name)} "
        f"VALUES ({placeholders})"
    )

    converted_rows = [
        [
            convert_value(value, data_type)
            for value, data_type in zip(row, data_types)
        ]
        for row in rows
    ]

    if converted_rows:
        connection.executemany(insert_sql, converted_rows)

    connection.commit()
    connection.execute("PRAGMA query_only = ON")

    metadata = {
        "file": str(csv_path),
        "table": table_name,
        "columns": list(zip(headers, data_types)),
        "row_count": len(rows),
        "sample_rows": rows[:3],
    }

    return connection, metadata


def build_schema_context(metadata: dict) -> str:
    columns = ", ".join(
        f"{name} ({data_type})"
        for name, data_type in metadata["columns"]
    )

    return (
        f"Table: {metadata['table']}\n"
        f"Source file: {metadata['file']}\n"
        f"Rows: {metadata['row_count']}\n"
        f"Columns: {columns}\n"
        f"Sample rows: {metadata['sample_rows']}"
    )


def strip_sql_fences(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(
        r"^```(?:sql)?\s*",
        "",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip()


def validate_sql(query: str) -> dict:
    cleaned = strip_sql_fences(query)

    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()

    if not cleaned:
        return {
            "valid": False,
            "reason": (
                "Gemini returned empty SQL (its output budget may have "
                "been used up by thinking tokens)."
            ),
        }

    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        return {
            "valid": False,
            "reason": "Only SELECT queries are permitted.",
        }

    if not re.search(r"\bSELECT\b", cleaned, re.IGNORECASE):
        return {
            "valid": False,
            "reason": "The query must contain SELECT.",
        }

    if ";" in cleaned:
        return {
            "valid": False,
            "reason": "Multiple SQL statements are not permitted.",
        }

    if re.search(r"(--|/\*|\*/)", cleaned):
        return {
            "valid": False,
            "reason": "SQL comments are not permitted.",
        }

    blocked_match = BLOCKED_KEYWORDS.search(cleaned)

    if blocked_match:
        return {
            "valid": False,
            "reason": (
                f"Blocked keyword: "
                f"{blocked_match.group(1).upper()}"
            ),
        }

    if "UNANSWERABLE" in cleaned.upper():
        return {
            "valid": False,
            "reason": (
                "The question cannot be answered from "
                "business_metrics.csv."
            ),
        }

    return {
        "valid": True,
        "reason": "OK",
        "sql": cleaned,
    }


def generate_sql(query: str, schema_context: str) -> dict:
    response = get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=(
            f"SCHEMA:\n{schema_context}\n\n"
            f"QUESTION:\n{query}"
        ),
        config=types.GenerateContentConfig(
            system_instruction=SQL_SYSTEM_INSTRUCTION,
            temperature=0.0,
            max_output_tokens=SQL_MAX_OUTPUT_TOKENS,
        ),
    )

    input_tokens, output_tokens = get_token_counts(response)

    return {
        "sql": strip_sql_fences(response.text or ""),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def sqlite_authorizer(action, _arg1, _arg2, _database, _source):
    blocked_actions = {
        getattr(sqlite3, name)
        for name in (
            "SQLITE_ATTACH",
            "SQLITE_DETACH",
            "SQLITE_DELETE",
            "SQLITE_DROP_INDEX",
            "SQLITE_DROP_TABLE",
            "SQLITE_DROP_TRIGGER",
            "SQLITE_DROP_VIEW",
            "SQLITE_INSERT",
            "SQLITE_PRAGMA",
            "SQLITE_TRANSACTION",
            "SQLITE_UPDATE",
        )
        if hasattr(sqlite3, name)
    }

    if action in blocked_actions:
        return sqlite3.SQLITE_DENY

    return sqlite3.SQLITE_OK


def execute_sql(
    connection: sqlite3.Connection,
    query: str,
) -> tuple[list[str], list[tuple], bool]:
    deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS

    def stop_long_query():
        return int(time.monotonic() >= deadline)

    connection.set_authorizer(sqlite_authorizer)
    connection.set_progress_handler(stop_long_query, 1_000)

    try:
        cursor = connection.execute(query)
        columns = [
            description[0]
            for description in cursor.description
        ]

        rows = cursor.fetchmany(MAX_QUERY_ROWS + 1)
        truncated = len(rows) > MAX_QUERY_ROWS

        return columns, rows[:MAX_QUERY_ROWS], truncated

    finally:
        connection.set_progress_handler(None, 0)
        connection.set_authorizer(None)


def interpret_results(
    question: str,
    sql: str,
    columns: list[str],
    rows: list[tuple],
    truncated: bool,
) -> dict:
    shown_rows = rows[:MAX_ROWS_TO_MODEL]
    shown_rows_truncated = len(rows) > MAX_ROWS_TO_MODEL

    response = get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=(
            f"USER QUESTION:\n{question}\n\n"
            f"SQL USED:\n{sql}\n\n"
            f"COLUMNS:\n{columns}\n\n"
            f"ROWS RETURNED:\n{len(rows)}\n\n"
            f"ROWS SHOWN:\n{shown_rows}\n\n"
            f"QUERY RESULT TRUNCATED:\n{truncated}\n"
            f"ROWS SHOWN TO MODEL TRUNCATED:\n"
            f"{shown_rows_truncated}"
        ),
        config=types.GenerateContentConfig(
            system_instruction=INTERPRET_SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=INTERPRET_MAX_OUTPUT_TOKENS,
        ),
    )

    input_tokens, output_tokens = get_token_counts(response)

    return {
        "answer": response.text or "Gemini returned no interpretation.",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def run(query: str) -> dict:
    connection = None
    sql_result = {
        "sql": "",
        "input_tokens": 0,
        "output_tokens": 0,
    }
    source_file = ""

    try:
        connection, metadata = load_csv_database()
        source_file = metadata["file"]

        schema_context = build_schema_context(metadata)
        sql_result = generate_sql(query, schema_context)
        validation = validate_sql(sql_result["sql"])

        if not validation["valid"]:
            return {
                "answer": f"Query blocked: {validation['reason']}",
                "sql": sql_result["sql"],
                "columns": [],
                "rows": [],
                "validation": "FAILED",
                "source_file": source_file,
                "input_tokens": sql_result["input_tokens"],
                "output_tokens": sql_result["output_tokens"],
            }

        columns, rows, truncated = execute_sql(
            connection,
            validation["sql"],
        )

        interpretation = interpret_results(
            query,
            validation["sql"],
            columns,
            rows,
            truncated,
        )

        return {
            "answer": interpretation["answer"],
            "sql": validation["sql"],
            "columns": columns,
            "rows": rows,
            "validation": "PASSED",
            "source_file": source_file,
            "input_tokens": (
                sql_result["input_tokens"]
                + interpretation["input_tokens"]
            ),
            "output_tokens": (
                sql_result["output_tokens"]
                + interpretation["output_tokens"]
            ),
        }

    except Exception as exc:
        return {
            "answer": f"Query execution failed: {exc}",
            "sql": sql_result["sql"],
            "columns": [],
            "rows": [],
            "validation": "ERROR",
            "source_file": source_file,
            "input_tokens": sql_result["input_tokens"],
            "output_tokens": sql_result["output_tokens"],
        }

    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    import sys

    # python quantitative_agent.py "your question"   (or no argument for the default test)
    test_query = (
        " ".join(sys.argv[1:])
        or "Show the first five rows from the business metrics data."
    )

    result = run(test_query)

    print("\n[Quantitative Agent Test]")
    print(f"Answer: {result['answer']}")
    print(f"SQL: {result['sql']}")
    print(f"Validation: {result['validation']}")
    print(f"Source file: {result['source_file']}")
    print(
        f"Tokens: input={result['input_tokens']} "
        f"output={result['output_tokens']}"
    )