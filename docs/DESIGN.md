# cura-eval design notes

## Goal

One OSS repo that (1) releases the medical evaluation harness used to evaluate Cura —
HealthBench (hard), HealthBench Professional, MedXpertQA text/MM, AgentClinic, and
MedAgentBench — in a **model-agnostic** form, and (2) shows how to integrate the Cura API
into local agents.

## The one seam

Every benchmark drives the model under test through `cura_eval.client.ChatClient`, a thin
bounded-concurrency, retrying wrapper over an OpenAI-compatible chat-completions endpoint.
Nothing else in the harness knows what serves the model. Two clients per run:

- **candidate** — the model under test (`--model` / `--base-url` / `--api-key-env`;
  defaults to the Cura API).
- **judge** — the grading/NPC endpoint for LLM-judged benchmarks (`--judge-*`; defaults to
  the official OpenAI API). Kept separate on purpose: comparisons are only valid when the
  judge is held fixed while candidates vary.

Reasoning traces are captured from `message.reasoning_content` (Cura/Kimi/DeepSeek) or
`message.reasoning` (OpenRouter) and persisted as `reasoning_content` on saved rows —
never silently dropped — so graded runs double as SFT-ready thinking data.

## Provenance

Ported from Actava's internal evaluation stack, with the training-framework coupling
(Tinker renderers/sampling) replaced by the OpenAI-compatible seam:

| Here | Origin | Notes |
|---|---|---|
| `judge.py` | internal `eval/judge.py` | upstream simple-evals HealthBench judge, verbatim template + scoring |
| `loaders.py` | internal `eval/loaders.py` | plus the regular/consensus HealthBench variants |
| `mcq.py` | internal MedXpertQA grader | deterministic, last-occurrence-wins letter extraction |
| `benchmarks/medagentbench/` | internal port of upstream MedAgentBench | graders + task data verbatim; agent loop rebuilt on native OpenAI function calling |
| `benchmarks/agentclinic/` | internal port of upstream AgentClinic | NPC prompts + episode control flow upstream-faithful; doctor rebuilt on the seam; scenario JSONL fetched from upstream at run time |

## Deliberate protocol choices

- **MedAgentBench uses native tool calling** (`fhir_get`/`fhir_post`/`finish` function
  specs) instead of upstream's strict GET/POST/FINISH text protocol. Rationale: the
  text protocol mostly measures format compliance; native tool calling measures the
  agentic capability current models actually ship with. Same tasks, same graders,
  different actuation — summaries carry a non-comparability note.
- **`fhir_post` is record-only** (upstream behavior): grading validates the payload
  structure from the tool-call record; nothing is written to the FHIR server.
- **AgentClinic doctor system prompt states a fixed question budget** (no per-turn
  count), so an episode has one stable system message. Image gating covers `NEJM_Ext`
  (upstream's exact `== "NEJM"` check misses it). Moderator/patient/measurement prompts
  are otherwise verbatim, including upstream's "corrent" typo — kept so grading matches
  upstream bit-for-bit.
- **Greedy decoding by default** (`temperature 0.0`) so a measurement is reproducible;
  `--temperature none` omits the field for endpoints that reject sampling params.
- **Judge failures never crash a run**: the example is recorded with
  `judge_error: true` and excluded from the headline mean (count reported).

## Output contract

Per run: `responses.jsonl` (per-example `{<id>, messages, response, score}` rows, full
conversation with `reasoning_content` lifted) + `summary.json` (headline score,
per-category breakdowns, run config). Row/score keys deliberately match Actava's internal
per-benchmark schemas so results interchange with existing tooling.

## Non-goals (v0.1)

- pass@k / multi-sample decoding, physician-baseline scoring for HBP, HealthBench
  consensus-specific metrics, MedAgentBench text-protocol compatibility mode, AgentClinic
  bias variants and MIMIC-IV dataset (needs credentialed access).
