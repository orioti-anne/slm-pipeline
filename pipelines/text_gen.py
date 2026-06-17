import json
import re
from typing import Generator

import numpy as np
import mlx.core as mx
from config import MODELS, ROUTER_SYSTEM, SYNTHESIS_SYSTEM
from core import model_manager

_CFG = MODELS["text_gen"]


def _make_repetition_penalty(penalty: float = 1.3, context_size: int = 64):
    """최근 생성된 토큰에 패널티를 부여해 반복 루프를 방지."""
    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        if tokens.size == 0:
            return logits
        recent = set(tokens[-context_size:].tolist())
        if not recent:
            return logits
        vocab_size = logits.shape[-1]
        mask_np = np.zeros(vocab_size, dtype=bool)
        for tid in recent:
            if 0 <= tid < vocab_size:
                mask_np[tid] = True
        mask = mx.array(mask_np)
        penalized = mx.where(logits > 0, logits / penalty, logits * penalty)
        return mx.where(mask, penalized, logits)
    return processor


def _load():
    if not model_manager.loaded("gen_model"):
        from mlx_lm import load
        model, tokenizer = load(_CFG["path"])
        model_manager.set("gen_model", model)
        model_manager.set("gen_tokenizer", tokenizer)


def _build_prompt(messages: list[dict]) -> str:
    tokenizer = model_manager.get("gen_tokenizer")
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Gemma 등 system role 미지원 모델: system 내용을 첫 user 메시지 앞에 합침
        merged = []
        system_text = ""
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            elif m["role"] == "user":
                content = f"{system_text}\n\n{m['content']}" if system_text else m["content"]
                merged.append({"role": "user", "content": content})
                system_text = ""
            else:
                merged.append(m)
        return tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=True
        )


def run(user_text: str, system: str = SYNTHESIS_SYSTEM) -> str:
    _load()
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    prompt = _build_prompt([
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ])
    with mx.stream(mx.gpu):
        return generate(
            model_manager.get("gen_model"),
            model_manager.get("gen_tokenizer"),
            prompt=prompt,
            max_tokens=_CFG["max_tokens"],
            sampler=make_sampler(temp=_CFG["temperature"], min_p=0.05),
            logits_processors=[_make_repetition_penalty(1.5, context_size=128)],
        ).strip()


def stream(user_text: str, system: str = SYNTHESIS_SYSTEM, max_tokens: int = None) -> Generator[str, None, None]:
    _load()
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt = _build_prompt([
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ])
    with mx.stream(mx.gpu):
        for chunk in stream_generate(
            model_manager.get("gen_model"),
            model_manager.get("gen_tokenizer"),
            prompt=prompt,
            max_tokens=max_tokens or _CFG["max_tokens"],
            sampler=make_sampler(temp=_CFG["temperature"], min_p=0.05),
            logits_processors=[_make_repetition_penalty(1.5, context_size=128)],
        ):
            yield chunk.text


def detect_pipelines(user_text: str) -> list[str]:
    """Router: Gemma로 필요한 파이프라인 목록을 JSON으로 반환."""
    _load()
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    prompt = _build_prompt([
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": user_text},
    ])
    with mx.stream(mx.gpu):
        raw = generate(
            model_manager.get("gen_model"),
            model_manager.get("gen_tokenizer"),
            prompt=prompt,
            max_tokens=_CFG["router_max_tokens"],
            sampler=make_sampler(temp=_CFG["router_temperature"]),
        )

    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            valid = {"text_gen", "translation", "sentiment", "ner", "qna"}
            pipelines = json.loads(match.group())
            return [p for p in pipelines if p in valid] or ["text_gen"]
        except (json.JSONDecodeError, TypeError):
            pass
    return ["text_gen"]
