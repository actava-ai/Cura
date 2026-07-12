"""MedAgentBench: multi-turn agentic FHIR benchmark over native OpenAI function calling.

Ported from MedAgentBench (Stanford ML Group, https://github.com/stanfordmlgroup/MedAgentBench).
The graders (``refsol.py``) and task data (``assets/``) are upstream; the agent protocol here
is native tool calling (``fhir_get`` / ``fhir_post`` / ``finish``) instead of upstream's strict
GET/POST/FINISH text protocol, so scores are NOT directly comparable to published numbers —
this measures a model's native tool-calling agentic ability on the same tasks and graders.
"""

from cura_eval.benchmarks.medagentbench.runner import run_medagentbench

__all__ = ["run_medagentbench"]
