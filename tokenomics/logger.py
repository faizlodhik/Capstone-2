
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "tokenomics_log.jsonl"

COST_PER_1K_INPUT = 0.003
COST_PER_1K_OUTPUT = 0.015


def log(
    query: str,
    agent: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict:
    input_tokens = max(0, int(input_tokens or 0))
    output_tokens = max(0, int(output_tokens or 0))

    input_cost = (input_tokens / 1000) * COST_PER_1K_INPUT
    output_cost = (output_tokens / 1000) * COST_PER_1K_OUTPUT
    total_cost = input_cost + output_cost

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "agent": agent,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": round(total_cost, 6),
        "cost_per_1000_queries": round(total_cost * 1000, 2),
    }

    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")

    print(
        f"[TOKENOMICS] Agent: {agent} | "
        f"Input: {input_tokens} | "
        f"Output: {output_tokens} | "
        f"Cost: ${total_cost:.6f}"
    )

    return entry
