"""Per-example row schema + run output writers.

Every benchmark writes the same two files per run:

- ``responses.jsonl`` — one row per example:
  ``{<id_key>: ..., "messages": [...], "response": ..., "score": {...}}``.
  ``messages`` is the entire conversation (input turns + the generated assistant turn,
  chain-of-thought lifted into ``reasoning_content``), so a graded run doubles as
  SFT-trainable thinking data.
- ``summary.json`` — aggregate metrics plus the exact run config for reproducibility.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def assistant_message(content: str, reasoning: str | None = None) -> dict[str, Any]:
    """One assistant turn; ``reasoning_content`` included only when non-empty."""
    if reasoning:
        return {"role": "assistant", "reasoning_content": reasoning, "content": content}
    return {"role": "assistant", "content": content}


def response_row(
    conversation: list[dict[str, Any]],
    assistant: dict[str, Any],
    *,
    example_id: str,
    id_key: str = "task_id",
    response: str | None = None,
    score: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The full re-trainable trajectory row (keys in id -> messages -> response -> score order)."""
    row: dict[str, Any] = {id_key: example_id, "messages": [*conversation, assistant]}
    if response is not None:
        row["response"] = response
    if score is not None:
        row["score"] = score
    return row


def strip_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace base64 image parts with a placeholder so saved rows stay small."""
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    parts.append({"type": "text", "text": "[image omitted]"})
                else:
                    parts.append(p)
            out.append({**m, "content": parts})
        else:
            out.append(m)
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def breakdown(pairs: list[tuple[str | None, float]]) -> dict[str, float]:
    """Mean value per non-empty group label, sorted by label."""
    sums: dict[str, list[float]] = {}
    for label, value in pairs:
        if label:
            sums.setdefault(label, []).append(value)
    return {k: mean(v) for k, v in sorted(sums.items())}


def run_stem(benchmark: str, model: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", model).strip("-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{benchmark}_{slug}_{stamp}"


def write_run(
    output_dir: str | Path,
    benchmark: str,
    model: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> Path:
    """Write ``responses.jsonl`` + ``summary.json`` under a fresh per-run directory."""
    run_dir = Path(output_dir) / run_stem(benchmark, model)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "responses.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return run_dir
