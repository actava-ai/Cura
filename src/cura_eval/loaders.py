"""Dataset loaders for the single-turn benchmarks (Hugging Face -> plain dataclasses).

All prompts are standard OpenAI chat messages; MedXpertQA-MM rows carry base64
``image_url`` parts, so any vision-capable OpenAI-compatible endpoint can consume them
unchanged. Override the local cache root with ``CURA_EVAL_CACHE``.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


def cache_root() -> Path:
    return Path(os.environ.get("CURA_EVAL_CACHE") or (Path.home() / ".cache" / "cura_eval"))


# --------------------------------------------------------------------------- HealthBench

# variant -> (HF repo, split). "hard" is the headline variant.
HB_VARIANTS = {
    "hard": ("neuralleap/healthbench-hard", "train"),
    "regular": ("neuralleap/healthbench-regular", "test"),
    "consensus": ("neuralleap/healthbench-consensus", "train"),
}


@dataclass(frozen=True)
class HBExample:
    task_id: str
    prompt: list[dict[str, str]]
    criteria: list[str]
    points: list[float]
    axes: list[str]
    theme: str


def _hb_theme(example: dict) -> str:
    for tag in example.get("example_tags", []):
        if tag.startswith("theme:"):
            return tag.split(":", 1)[1]
    return ""


def _hb_parse(example: dict) -> HBExample:
    criteria: list[str] = []
    points: list[float] = []
    axes: list[str] = []
    for rubric in example.get("rubrics", []):
        criteria.append(rubric["criterion"])
        points.append(rubric["points"])
        tags = {k: v for k, _, v in (t.partition(":") for t in rubric.get("tags", [])) if k}
        axes.append(tags.get("axis", "unknown"))
    prompt = example.get("prompt", [])
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    return HBExample(str(example["prompt_id"]), prompt, criteria, points, axes, _hb_theme(example))


def load_healthbench(variant: str = "hard", limit: int | None = None) -> list[HBExample]:
    from datasets import load_dataset

    if variant not in HB_VARIANTS:
        raise ValueError(
            f"unknown HealthBench variant {variant!r}; expected one of {sorted(HB_VARIANTS)}"
        )
    repo, split = HB_VARIANTS[variant]
    ds = load_dataset(repo, split=split)
    rows = ds if limit is None else ds.select(range(min(limit, len(ds))))
    return [_hb_parse(dict(r)) for r in rows]


# --------------------------------------------------------------------- HealthBench Professional

HBP_REPO, HBP_SPLIT = "openai/healthbench-professional", "test"


@dataclass(frozen=True)
class HBPExample:
    task_id: str
    conversation: list[dict[str, str]]
    rubric_items: list[dict[str, Any]]
    use_case: str
    type: str
    difficulty: str
    specialty: str


def _hbp_parse(example: dict) -> HBPExample:
    return HBPExample(
        task_id=str(example["id"]),
        conversation=example["conversation"]["messages"],
        rubric_items=list(example.get("rubric_items", [])),
        use_case=example["use_case"],
        type=example["type"],
        difficulty=example["difficulty"],
        specialty=example["specialty"],
    )


def load_hbp(limit: int | None = None) -> list[HBPExample]:
    from datasets import load_dataset

    ds = load_dataset(HBP_REPO, split=HBP_SPLIT)
    rows = ds if limit is None else ds.select(range(min(limit, len(ds))))
    return [_hbp_parse(dict(r)) for r in rows]


# --------------------------------------------------------------------------------- MedXpertQA

MXQ_REPO = "TsinghuaC3I/MedXpertQA"
_MXQ_CONFIG = {"text": "Text", "mm": "MM"}


@dataclass(frozen=True)
class MXQExample:
    task_id: str
    messages: list[dict[str, Any]]  # OpenAI-style; mm has base64 image_url parts
    answer: str
    valid_letters: str
    medical_task: str | None
    body_system: str | None


def _options_block(options: dict[str, str]) -> str:
    return "\n".join(f"{k}. {options[k]}" for k in sorted(options))


def _mxq_prompt(question: str, options: dict[str, str], max_letter: str, *, mm: bool) -> str:
    ref = (
        "The image(s) attached to this message are referenced in the question above. " if mm else ""
    )
    return (
        f"{question}\n\n{_options_block(options)}\n\n"
        f"{ref}Answer with a single option letter from A through {max_letter}. "
        f"Put your final answer inside \\boxed{{X}}."
    )


def _b64_image_part(path: Path) -> dict:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def _ensure_images_dir() -> Path:
    images_cache = cache_root() / "medxpertqa_images"
    marker = images_cache / ".extracted"
    if marker.exists():
        return images_cache
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(MXQ_REPO, "images.zip", repo_type="dataset")
    images_cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(images_cache)
    marker.touch()
    return images_cache


def _resolve_image(root: Path, filename: str) -> Path:
    target = Path(filename).name
    for f in Path(root).rglob(target):
        if f.is_file():
            return f
    raise FileNotFoundError(f"image {target!r} not found under {root}")


def _extract_images(raw: Any) -> list[str]:
    if not raw:
        return []
    names: list[str] = []
    for item in raw:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = item.get("filename") or item.get("path") or item.get("name")
            if name:
                names.append(str(name))
    return names


def load_medxpertqa(subset: Literal["text", "mm"], limit: int | None = None) -> list[MXQExample]:
    from datasets import load_dataset

    ds = load_dataset(MXQ_REPO, name=_MXQ_CONFIG[subset], split="test")
    rows = ds if limit is None else ds.select(range(min(limit, len(ds))))
    images_root = _ensure_images_dir() if subset == "mm" else None
    out: list[MXQExample] = []
    for r in rows:
        r = dict(r)
        options = dict(r.get("options") or {})
        valid_letters = "".join(sorted(o.upper() for o in options)) or "ABCDEFGHIJ"
        max_letter = valid_letters[-1]
        answer = str(r["label"]).strip().upper()
        images = _extract_images(r.get("images"))
        if subset == "mm":
            if not images:
                raise ValueError(f"MM row {r['id']!r} has no images")
            assert images_root is not None
            text_part = {
                "type": "text",
                "text": _mxq_prompt(r["question"], options, max_letter, mm=True),
            }
            image_parts = [_b64_image_part(_resolve_image(images_root, n)) for n in images]
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": [text_part, *image_parts]}
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": _mxq_prompt(r["question"], options, max_letter, mm=False),
                }
            ]
        out.append(
            MXQExample(
                task_id=str(r["id"]),
                messages=messages,
                answer=answer,
                valid_letters=valid_letters,
                medical_task=r.get("medical_task"),
                body_system=r.get("body_system"),
            )
        )
    return out
