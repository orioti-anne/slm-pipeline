import re


_SYSTEM = (
    "주어진 한국어 문장의 감성을 분석하세요. "
    "반드시 아래 형식으로만 답하세요: LABEL SCORE\n"
    "LABEL은 긍정, 부정, 중립 중 하나. SCORE는 0~100 정수.\n"
    "예: 긍정 85"
)


def run(text: str) -> dict:
    import pipelines.text_gen as text_gen
    text_gen._load()

    raw = text_gen.run(text, system=_SYSTEM)

    match = re.search(r'(긍정|부정|중립)[^\d]*(\d+)', raw)
    if match:
        label = match.group(1)
        score = round(int(match.group(2)) / 100, 4)
    else:
        label = "중립"
        score = 0.5

    return {"label": label, "score": score}
