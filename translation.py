from config import MODELS
from core import model_manager

_CFG = MODELS["translation"]


def _load_ko_en():
    if not model_manager.loaded("translation_ko_en"):
        from transformers import MarianMTModel, MarianTokenizer
        tokenizer = MarianTokenizer.from_pretrained(_CFG["ko_to_en"])
        model = MarianMTModel.from_pretrained(_CFG["ko_to_en"])
        model_manager.set("translation_ko_en", (tokenizer, model))
    return model_manager.get("translation_ko_en")


def _load_en_ko():
    if not model_manager.loaded("translation_en_ko"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(_CFG["en_to_ko"], src_lang="eng_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(_CFG["en_to_ko"])
        model_manager.set("translation_en_ko", (tokenizer, model))
    return model_manager.get("translation_en_ko")


def run(text: str, direction: str = "ko_en") -> dict:
    """
    direction: "ko_en" (한→영) or "en_ko" (영→한)
    """
    if direction == "ko_en":
        tokenizer, model = _load_ko_en()
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elif direction == "en_ko":
        tokenizer, model = _load_en_ko()
        inputs = tokenizer(text, return_tensors="pt")
        target_lang_id = tokenizer.convert_tokens_to_ids("kor_Hang")
        outputs = model.generate(**inputs, forced_bos_token_id=target_lang_id, max_new_tokens=256)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        return {"error": f"Invalid direction: {direction}"}

    return {"translated": result, "direction": direction}
