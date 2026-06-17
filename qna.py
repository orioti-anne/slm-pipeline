import torch
from config import MODELS
from core import model_manager

_CFG = MODELS["qna"]


def _load():
    if not model_manager.loaded("qna"):
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        tokenizer = AutoTokenizer.from_pretrained(_CFG["path"])
        model = AutoModelForQuestionAnswering.from_pretrained(_CFG["path"])
        model_manager.set("qna", (tokenizer, model))


def _sliding_window_chunks(tokenizer, question: str, context: str,
                            max_length: int = 512, stride: int = 128):
    """context를 슬라이딩 윈도우로 분할해 (chunk_text, offset) 리스트 반환."""
    q_ids = tokenizer.encode(question, add_special_tokens=False)
    # [CLS] q [SEP] ctx [SEP] = 3 special tokens
    max_ctx_tokens = max_length - len(q_ids) - 3

    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(ctx_ids):
        end = min(start + max_ctx_tokens, len(ctx_ids))
        chunk_text = tokenizer.decode(ctx_ids[start:end], skip_special_tokens=True)
        chunks.append(chunk_text)
        if end >= len(ctx_ids):
            break
        start += max_ctx_tokens - stride

    return chunks


def run(question: str, context: str = "") -> dict:
    _load()
    if not context:
        return {"answer": None, "error": "context required for QnA pipeline"}

    tokenizer, model = model_manager.get("qna")

    chunks = _sliding_window_chunks(tokenizer, question, context)
    best_answer = None
    best_chunk = None
    best_score = -1.0

    for chunk in chunks:
        inputs = tokenizer(question, chunk, return_tensors="pt",
                           truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        start_idx = int(torch.argmax(outputs.start_logits))
        end_idx = int(torch.argmax(outputs.end_logits)) + 1

        if end_idx <= start_idx:
            continue

        answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        score = float(torch.softmax(outputs.start_logits, dim=-1).max())

        # 너무 짧거나 숫자만인 스팬은 의미없는 추출로 제외
        if not answer or len(answer) < 2 or answer.strip().isdigit():
            continue

        if score > best_score:
            best_score = score
            best_answer = answer
            best_chunk = chunk

    # 신뢰도가 너무 낮으면 답변 없음 처리
    if best_score < 0.2:
        best_answer = None

    return {
        "answer": best_answer,
        "score": round(best_score, 4) if best_score >= 0 else 0.0,
        "chunk": best_chunk,
    }
