from config import DOC_QA_SYSTEM


def run(question: str, context: str = "") -> dict:
    import pipelines.text_gen as text_gen
    text_gen._load()

    if not context:
        return {"answer": None, "error": "context required for QnA pipeline"}

    # 문서가 너무 길면 앞 3000자만 사용
    ctx = context[:3000]
    prompt = f"[문서 발췌]\n{ctx}\n\n[질문]\n{question}"
    answer = text_gen.run(prompt, system=DOC_QA_SYSTEM)

    return {"answer": answer.strip() if answer else None, "score": 1.0}
