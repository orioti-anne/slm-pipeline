from config import MODELS
from core import model_manager

_CFG = MODELS["ner"]

_TYPE_KO = {
    "PER": "인물", "LOC": "장소", "ORG": "기관",
    "DAT": "날짜", "TIM": "시간", "NUM": "수량",
    "CVL": "문명", "AFW": "인공물", "ANM": "동물",
    "PLT": "식물", "MAT": "물질", "TRM": "용어",
    "EVN": "사건", "FLD": "분야", "POH": "기타",
}


def _load():
    if not model_manager.loaded("ner"):
        from transformers import pipeline
        pipe = pipeline(
            "token-classification",
            model=_CFG["path"],
            aggregation_strategy="none",
        )
        model_manager.set("ner", pipe)


def _group_entities(tokens: list, original_text: str) -> list:
    entities = []
    current = None

    for t in tokens:
        label = t["entity"]
        tag_type, bio = label.rsplit("-", 1)

        if bio == "B":
            if current:
                entities.append(current)
            current = {
                "type": tag_type,
                "start": t["start"],
                "end": t["end"],
                "score": float(t["score"]),
                "count": 1,
            }
        elif bio == "I" and current and current["type"] == tag_type:
            current["end"] = t["end"]
            current["score"] += float(t["score"])
            current["count"] += 1
        else:
            if current:
                entities.append(current)
            current = None

    if current:
        entities.append(current)

    result = []
    for e in entities:
        text = original_text[e["start"]:e["end"]]
        result.append({
            "text": text,
            "type": _TYPE_KO.get(e["type"], e["type"]),
            "score": round(e["score"] / e["count"], 4),
        })
    return result


def run(text: str) -> dict:
    _load()
    pipe = model_manager.get("ner")
    tokens = pipe(text)
    entities = _group_entities(tokens, text)
    return {"entities": entities}
