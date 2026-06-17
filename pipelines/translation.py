import re
from core import model_manager

_KO_EN_SYSTEM = "You are a professional Korean-English translator. Translate the given Korean text into natural English. Output only the translated text, nothing else."
_EN_KO_SYSTEM = "You are a professional English-Korean translator. Translate the given English text into natural Korean. Use standard Korean names for proper nouns (e.g. Suwon→수원, Samsung→삼성, Seoul→서울). Output only the translated text, nothing else."


def _strip_instruction(text: str) -> str:
    text = text.strip()
    cleaned = re.sub(
        r'[\s,]*(?:이것을|이걸|아래를|다음을|[을를])?[\s]*(?:영어|한국어|한글|영문|국문)로\s*(?:번역해\s*(?:줘|주세요|주셔요|주시오)|번역하세요|번역해)[.!?]*\s*$',
        '', text
    ).strip()
    return cleaned if cleaned else text


def run(text: str, direction: str = "ko_en") -> dict:
    import pipelines.text_gen as text_gen
    text_gen._load()

    src = _strip_instruction(text)
    system = _KO_EN_SYSTEM if direction == "ko_en" else _EN_KO_SYSTEM
    result = text_gen.run(src, system=system)
    return {"translated": result, "direction": direction}
