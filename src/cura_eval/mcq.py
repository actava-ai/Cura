"""MCQ letter extraction — pure scoring core (stdlib-only).

Tolerant letter extraction from free-form reasoning with last-occurrence-wins priority
``\\boxed{X}`` > ``Answer: X`` > standalone letter, and a dynamic ``[A..max_letter]``
range so a 5-option question never accepts letters F-J leaking in from reasoning prose.
"""

from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", flags=re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Drop ``<think>...</think>`` / ``<thinking>...</thinking>`` blocks."""
    return _THINK_BLOCK.sub(" ", text)


def extract_letter(text: str, max_letter: str = "J") -> str | None:
    """Extract the chosen option letter from free-form answer text."""
    if not text:
        return None
    max_letter = (max_letter or "J").upper()
    if not ("A" <= max_letter <= "Z"):
        max_letter = "J"
    cls = f"[A-{max_letter}]"

    cleaned = _strip_think(text).upper()
    if not cleaned.strip():
        return None

    boxed = re.findall(rf"\\?BOXED\s*\{{\s*\(?\s*({cls})\s*\)?\s*\}}", cleaned)
    if boxed:
        return boxed[-1]

    marker = re.findall(rf"(?:FINAL\s+)?ANSWER\s*(?:IS\s+)?[:=]?\s*\(?({cls})\)?\b", cleaned)
    if marker:
        return marker[-1]

    standalone = re.findall(rf"\b({cls})\b", cleaned)
    if standalone:
        return standalone[-1]
    return None


def decide(answer_text: str, gold_label: str, max_letter: str = "J") -> bool:
    """True iff the extracted letter exactly matches the gold label."""
    chosen = extract_letter(answer_text, max_letter)
    return chosen is not None and chosen == gold_label.strip().upper()
