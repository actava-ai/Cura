"""Evaluate a model on MedXpertQA-Text programmatically.

Usage:
    export ACTAVA_API_KEY=...
    python examples/quickstart.py

Swap base_url/model/api_key_env to evaluate any other OpenAI-compatible endpoint.
"""

import asyncio
import json

from cura_eval import ChatClient
from cura_eval.benchmarks.medxpertqa import run_medxpertqa


async def main() -> None:
    candidate = ChatClient(
        model="actava/cura-soar",
        base_url="https://inference.actava.ai/v1",
        api_key_env="ACTAVA_API_KEY",
        concurrency=8,
    )
    rows, summary = await run_medxpertqa(candidate, subset="text", limit=10)
    print(json.dumps(summary, indent=2))
    for row in rows[:3]:
        print(row["task_id"], "->", row["score"]["chosen"], "gold:", row["score"]["gold"])


if __name__ == "__main__":
    asyncio.run(main())
