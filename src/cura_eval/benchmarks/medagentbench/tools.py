"""The three FHIR agent tools as OpenAI function-calling specs + their executor.

``fhir_get`` reads from the server; ``fhir_post`` is record-only (upstream never sends
POSTs — grading reads the (url, payload) pairs back out of the tool-call record);
``finish`` ends the episode with the final answer array.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from cura_eval.benchmarks.medagentbench import fhir

MAX_OBS_CHARS = 30000  # bound a large FHIR bundle per turn; tune for the model's context window

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fhir_get",
            "description": "Send a FHIR GET (search/read) request and return the JSON response body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full FHIR GET URL including query params, built on the api_base.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fhir_post",
            "description": "Create a FHIR resource via POST. Returns an acknowledgement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full FHIR POST URL built on the api_base.",
                    },
                    "payload": {
                        "type": "object",
                        "description": "The FHIR resource JSON object to create.",
                    },
                },
                "required": ["url", "payload"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Submit the final answer and end the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "array",
                        "description": (
                            "Final answer values in requested order, "
                            'e.g. [6.2, "2023-11-09T00:17:00+00:00"].'
                        ),
                        "items": {},
                    }
                },
                "required": ["answer"],
            },
        },
    },
]


def _ensure_format(url: str) -> str:
    if "_format=" in url:
        return url
    return url + ("&" if "?" in url else "?") + "_format=json"


@dataclass
class EpisodeState:
    """What grading needs from an episode: the finish answer + every recorded POST.

    Exposes ``.result`` / ``.posts`` — the exact view the ported ``refsol`` graders consume.
    """

    result: str = "[]"  # json-encoded finish answer array ("[]" if finish was never called)
    posts: list[tuple[str, dict]] = field(default_factory=list)
    finished: bool = False


async def execute_tool_call(tool_call: dict[str, Any], state: EpisodeState, api_base: str) -> str:
    """Execute one OpenAI-shaped tool call, mutating ``state``; returns the observation text."""
    name = tool_call["function"]["name"]
    try:
        args = json.loads(tool_call["function"]["arguments"] or "{}")
    except Exception:
        return f"Error: arguments for {name} were not valid JSON."
    if not isinstance(args, dict):
        return f"Error: arguments for {name} must be a JSON object."

    if name == "fhir_get":
        res = await asyncio.to_thread(
            fhir.send_get_request, _ensure_format(str(args.get("url", "")))
        )
        if "data" in res:
            data = res["data"]
            content = data if isinstance(data, str) else json.dumps(data)
            return content[:MAX_OBS_CHARS]
        return f"Error in sending the GET request: {res.get('error')}"

    if name == "fhir_post":
        # Record-only: upstream never sends POSTs; grading validates (url, payload) structure.
        url, payload = args.get("url"), args.get("payload")
        if url is not None and payload is not None:
            state.posts.append((url, payload))
        return "POST request accepted and executed successfully."

    if name == "finish":
        state.result = json.dumps(args.get("answer", []))
        state.finished = True
        return state.result

    return f"Error: unknown tool {name!r}. Available tools: fhir_get, fhir_post, finish."
