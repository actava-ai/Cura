"""cura-eval CLI.

Defaults target the Cura API (https://inference.actava.ai/v1, model ``actava/cura-soar``,
key in ``ACTAVA_API_KEY``); point ``--base-url`` / ``--model`` / ``--api-key-env`` at any
other OpenAI-compatible endpoint to evaluate a different model. LLM-judged benchmarks
additionally need a judge endpoint (defaults to OpenAI via ``OPENAI_API_KEY``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from cura_eval import judge as judge_mod
from cura_eval.client import ChatClient

CURA_BASE_URL = "https://inference.actava.ai/v1"
CURA_MODEL = "actava/cura-soar"
CURA_KEY_ENV = "ACTAVA_API_KEY"


def _temperature(value: str) -> float | None:
    """``--temperature none`` omits the field (for endpoints that reject it)."""
    if value.strip().lower() in ("none", "null", "omit"):
        return None
    return float(value)


def _candidate_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("model under test (any OpenAI-compatible endpoint)")
    g.add_argument("--model", default=CURA_MODEL, help=f"model id (default: {CURA_MODEL})")
    g.add_argument(
        "--base-url",
        default=CURA_BASE_URL,
        help=f"OpenAI-compatible base URL (default: {CURA_BASE_URL}; "
        "use e.g. http://localhost:8000/v1 for vLLM)",
    )
    g.add_argument(
        "--api-key-env",
        default=CURA_KEY_ENV,
        help=f"env var holding the API key (default: {CURA_KEY_ENV}; "
        "use OPENAI_API_KEY / OPENROUTER_API_KEY / … for other providers)",
    )
    g.add_argument(
        "--temperature",
        type=_temperature,
        default=0.0,
        help="sampling temperature (default 0.0 for reproducible measurement; "
        "'none' omits the field for endpoints that reject it)",
    )
    g.add_argument("--max-tokens", type=int, default=None, help="per-request generation budget")
    g.add_argument("--concurrency", type=int, default=None, help="max in-flight examples")
    g.add_argument("--max-retries", type=int, default=3)
    g.add_argument(
        "--extra-body",
        default=None,
        help="JSON forwarded verbatim on every request "
        '(e.g. \'{"thinking": {"type": "enabled"}}\' or OpenRouter routing)',
    )
    g.add_argument("--limit", type=int, default=None, help="evaluate only the first N examples")
    g.add_argument("--output-dir", default="results")
    return p


def _judge_parser(default_model: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("judge endpoint (also OpenAI-compatible)")
    g.add_argument("--judge-model", default=default_model)
    g.add_argument("--judge-base-url", default=None, help="default: the official OpenAI API")
    g.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    return p


def _build_candidate(args: argparse.Namespace, *, default_concurrency: int) -> ChatClient:
    extra_body = json.loads(args.extra_body) if args.extra_body else None
    return ChatClient(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        concurrency=args.concurrency or default_concurrency,
        max_retries=args.max_retries,
        extra_body=extra_body,
    )


def _build_judge(args: argparse.Namespace) -> Any:
    return judge_mod.make_judge_client(
        base_url=args.judge_base_url, api_key_env=args.judge_api_key_env
    )


def _finish(args: argparse.Namespace, benchmark: str, rows: list[dict], summary: dict) -> None:
    from cura_eval.rows import write_run

    summary["config"] = {
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "limit": args.limit,
        "judge_model": getattr(args, "judge_model", None),
        "judge_base_url": getattr(args, "judge_base_url", None),
    }
    run_dir = write_run(args.output_dir, benchmark, args.model, rows, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {len(rows)} rows to {run_dir}/responses.jsonl", file=sys.stderr)


# --------------------------------------------------------------------------- subcommands


def _cmd_healthbench(args: argparse.Namespace) -> None:
    from cura_eval.benchmarks.healthbench import run_healthbench

    candidate = _build_candidate(args, default_concurrency=16)
    rows, summary = asyncio.run(
        run_healthbench(
            candidate,
            judge_client=_build_judge(args),
            judge_model=args.judge_model,
            variant=args.variant,
            limit=args.limit,
            max_tokens=args.max_tokens or 8192,
            temperature=args.temperature,
            concurrency=args.concurrency or 16,
        )
    )
    _finish(args, f"healthbench_{args.variant}", rows, summary)


def _cmd_hbp(args: argparse.Namespace) -> None:
    from cura_eval.benchmarks.healthbench_professional import run_healthbench_professional

    candidate = _build_candidate(args, default_concurrency=16)
    rows, summary = asyncio.run(
        run_healthbench_professional(
            candidate,
            judge_client=_build_judge(args),
            judge_model=args.judge_model,
            judge_reasoning_effort=args.judge_reasoning_effort,
            limit=args.limit,
            max_tokens=args.max_tokens or 32000,
            temperature=args.temperature,
            concurrency=args.concurrency or 16,
        )
    )
    _finish(args, "healthbench_professional", rows, summary)


def _cmd_medxpertqa(args: argparse.Namespace) -> None:
    from cura_eval.benchmarks.medxpertqa import run_medxpertqa

    candidate = _build_candidate(args, default_concurrency=32)
    rows, summary = asyncio.run(
        run_medxpertqa(
            candidate,
            subset=args.subset,
            limit=args.limit,
            max_tokens=args.max_tokens or 8192,
            temperature=args.temperature,
            concurrency=args.concurrency or 32,
        )
    )
    _finish(args, f"medxpertqa_{args.subset}", rows, summary)


def _cmd_medagentbench(args: argparse.Namespace) -> None:
    from cura_eval.benchmarks.medagentbench import run_medagentbench

    candidate = _build_candidate(args, default_concurrency=8)
    rows, summary = asyncio.run(
        run_medagentbench(
            candidate,
            fhir_api_base=args.fhir_api_base,
            version=args.version,
            limit=args.limit,
            max_round=args.max_round,
            max_tokens=args.max_tokens or 8192,
            temperature=args.temperature,
            concurrency=args.concurrency or 8,
        )
    )
    _finish(args, "medagentbench", rows, summary)


def _cmd_agentclinic(args: argparse.Namespace) -> None:
    from cura_eval.benchmarks.agentclinic import run_agentclinic

    candidate = _build_candidate(args, default_concurrency=8)
    rows, summary = asyncio.run(
        run_agentclinic(
            candidate,
            judge_client=_build_judge(args),
            judge_model=args.judge_model,
            dataset=args.dataset,
            limit=args.limit,
            total_inferences=args.total_inferences,
            max_tokens=args.max_tokens or 8192,
            temperature=args.temperature,
            concurrency=args.concurrency or 8,
            image_request=args.image_request,
            preserve_thinking=args.preserve_thinking,
        )
    )
    _finish(args, f"agentclinic_{args.dataset}", rows, summary)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cura-eval",
        description="Model-agnostic medical evaluation harness "
        "(works with any OpenAI-compatible endpoint).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    candidate = _candidate_parser()

    p = sub.add_parser(
        "healthbench",
        parents=[candidate, _judge_parser(judge_mod.HB_DEFAULT_JUDGE)],
        help="HealthBench (rubric-judged health conversations)",
    )
    p.add_argument("--variant", choices=["hard", "regular", "consensus"], default="hard")
    p.set_defaults(func=_cmd_healthbench)

    p = sub.add_parser(
        "healthbench-professional",
        parents=[candidate, _judge_parser(judge_mod.HBP_DEFAULT_JUDGE)],
        help="HealthBench Professional (rubric-judged, length-adjusted)",
    )
    p.add_argument("--judge-reasoning-effort", default="low")
    p.set_defaults(func=_cmd_hbp)

    p = sub.add_parser(
        "medxpertqa",
        parents=[candidate],
        help="MedXpertQA MCQ (judge-free; mm needs a vision endpoint)",
    )
    p.add_argument("--subset", choices=["text", "mm"], default="text")
    p.set_defaults(func=_cmd_medxpertqa)

    p = sub.add_parser(
        "medagentbench",
        parents=[candidate],
        help="MedAgentBench agentic FHIR tasks (needs the FHIR docker server; judge-free)",
    )
    p.add_argument(
        "--fhir-api-base",
        default="http://localhost:8080/fhir/",
        help="FHIR server base URL (docker run -p 8080:8080 jyxsu6/medagentbench:latest)",
    )
    p.add_argument("--version", choices=["v1", "v2"], default="v1")
    p.add_argument("--max-round", type=int, default=8, help="max assistant turns per task")
    p.set_defaults(func=_cmd_medagentbench)

    p = sub.add_parser(
        "agentclinic",
        parents=[candidate, _judge_parser("gpt-5.5")],
        help="AgentClinic simulated consultations (NPCs + moderator on the judge endpoint)",
    )
    p.add_argument("--dataset", choices=["MedQA", "MedQA_Ext", "NEJM", "NEJM_Ext"], default="MedQA")
    p.add_argument("--total-inferences", type=int, default=20, help="doctor question budget")
    p.add_argument(
        "--image-request",
        action="store_true",
        help="NEJM: doctor must say REQUEST IMAGES to see the case image "
        "(default: image attached every turn)",
    )
    p.add_argument(
        "--preserve-thinking",
        action="store_true",
        help="re-send prior-turn reasoning_content (endpoints with preserved thinking, "
        'e.g. Cura thinking keep:"all")',
    )
    p.set_defaults(func=_cmd_agentclinic)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
