"""Load MedAgentBench tasks + build the native-tool agent prompt.

Task data is vendored under ``assets/`` (upstream ``MedAgentBench/data/medagentbench/``);
the funcs JSON is embedded as in-prompt FHIR API documentation (the tools are generic
URL-based, so the model still needs to know the valid resources/queries).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_ASSETS = Path(__file__).parent / "assets"

# Native-tool system prompt. The model acts through the fhir_get/fhir_post/finish tools, so the
# upstream "output exactly GET/POST/FINISH as text" rules are intentionally dropped. The funcs
# JSON documents the available FHIR endpoints; {api_base} stays literal inside it (the model is
# told to substitute it). NOTE: json.dumps(funcs) is inserted as a *value*, so str.format never
# reprocesses the JSON's own braces — only the template's {api_base}/{functions} fields.
_SYSTEM_PROMPT_TEMPLATE = """You are an expert in using FHIR functions to assist medical professionals. \
You are given a clinical question and a set of FHIR API functions, exposed to you as the tools \
`fhir_get`, `fhir_post`, and `finish`.

- Call `fhir_get(url)` to read data and `fhir_post(url, payload)` to create a resource. Use \
{api_base} as the base of every FHIR URL.
- You may make multiple tool calls across turns. When you have completed all requested tasks, call \
`finish(answer)` exactly once. `answer` MUST be a JSON array containing ONLY the final answer value(s) \
in the requested order — and nothing else. STRICT rules for the array:
  - Include ONLY the value(s) the question asks for. No explanations, notes, reasoning, labels, status \
text, or units, and no extra elements (do not append a timestamp, dose, or sentence unless the \
question explicitly asks for it).
  - Numbers must be bare and unquoted: 6.2, not "6.2" or "6.2 mg/dL".
  - If you checked a value and then decided NOT to act on it (e.g. a level is not low so no order is \
needed), STILL return just that checked value, e.g. [2.1] — do NOT replace it with a sentence \
describing your decision.
  - If the task only asks you to act (create a resource) and has no value to report, call `finish([])`.
  - The array must be a flat list of values, never a JSON object or key/value pairs.
  Examples: finish([6.2]), finish(["S6534835"]), finish([]), finish([6.2, "2023-11-09T00:17:00+00:00"]).

Here is the list of available FHIR functions in JSON (use {api_base} as the api_base):
{functions}"""


@dataclass(frozen=True)
class MABExample:
    id: str
    instruction: str
    context: str
    sol: list | None
    eval_MRN: str
    raw: dict

    @property
    def task_type(self) -> str:
        return self.id.split("_")[0]


def _load_funcs() -> list[dict]:
    return json.loads((_ASSETS / "funcs_v1.json").read_text())


def load_medagentbench(version: str = "v1", max_examples: int | None = None) -> list[MABExample]:
    path = _ASSETS / f"test_data_{version}.json"
    if not path.exists():
        raise FileNotFoundError(f"unknown MedAgentBench data version {version!r} (no {path})")
    raw = json.loads(path.read_text())
    if max_examples is not None:
        raw = raw[:max_examples]
    return [
        MABExample(
            id=t["id"],
            instruction=t["instruction"],
            context=t.get("context", ""),
            sol=t.get("sol"),
            eval_MRN=t.get("eval_MRN", ""),
            raw=t,
        )
        for t in raw
    ]


def build_system_prompt(api_base: str) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(api_base=api_base, functions=json.dumps(_load_funcs()))


def build_user_message(ex: MABExample) -> str:
    if ex.context:
        return f"Context: {ex.context}\nQuestion: {ex.instruction}"
    return f"Question: {ex.instruction}"
