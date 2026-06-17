PORT = 8004
HOST = "0.0.0.0"

MODELS = {
    "text_gen": {
        "path": "mlx-community/gemma-2-2b-it-4bit",  # 텍스트 생성·번역·감성·QnA 통합
        "max_tokens": 100,
        "temperature": 0.7,
        "router_temperature": 0.0,
        "router_max_tokens": 60,
    },
    "ner": {
        "path": "monologg/koelectra-base-v3-naver-ner",  # 개체명 인식 전용
        "aggregation": "simple",
    },
}

ROUTER_SYSTEM = """You are a task router. Analyze the user input and return a JSON array of required pipeline tasks.

Available tasks:
- "text_gen": generating a conversational response
- "translation": translating between Korean and English
- "sentiment": analyzing emotion or feeling in the text
- "ner": extracting names, places, organizations, dates from text
- "qna": answering a question based on provided context/document

Rules:
- Always include "text_gen" when a natural language response is needed
- Return ONLY a valid JSON array, nothing else

Examples:
Input: "오늘 기분이 너무 안좋아, 서울 빵집 추천해줘" → ["sentiment", "text_gen"]
Input: "이 문장을 영어로 번역해줘" → ["translation"]
Input: "삼성전자의 본사는 수원에 있습니다. 영어로 번역해줘" → ["translation"]
Input: "안녕하세요, 만나서 반갑습니다. 영어로 번역해줘" → ["translation"]
Input: "Samsung Electronics is in Suwon. 한국어로 번역해줘" → ["translation"]
Input: "Hello, how are you? 한글로 번역해줘" → ["translation"]
Input: "서울에서 가장 유명한 산은?" → ["text_gen"]
Input: "안녕하세요!" → ["text_gen"]
Input: "삼성전자는 수원에 본사가 있다" → ["ner", "text_gen"]
Input: "애플의 CEO는 팀 쿡이다" → ["ner", "text_gen"]
Input: "이순신은 조선의 장군이다" → ["ner", "text_gen"]
Input: "현대자동차는 서울에 본사를 두고 있다" → ["ner", "text_gen"]
Input: "삼성전자는 수원에 본사를 두고 있고 현대자동차는 서울에 있다" → ["ner", "text_gen"]
Input: "대한민국 수도는 서울이고 미국 수도는 워싱턴이다" → ["ner", "text_gen"]
Input: "2024년 파리 올림픽에서 대한민국이 금메달을 땄다" → ["ner", "text_gen"]
Input: "카카오의 창업자는 김범수다" → ["ner", "text_gen"]
Input: "제주도는 대한민국의 남쪽에 위치한 섬이다" → ["ner", "text_gen"]
Input: "오늘 날씨가 우울해" → ["sentiment", "text_gen"]
Input: "퇴근 시간이 가까워져서 기분이 좋아져" → ["sentiment", "text_gen"]
Input: "시험에 합격해서 너무 기뻐" → ["sentiment", "text_gen"]
Input: "오늘 일이 너무 힘들었어" → ["sentiment", "text_gen"]
Input: "출근하기 싫어" → ["sentiment", "text_gen"]
Input: "집에 가고 싶어" → ["sentiment", "text_gen"]
Input: "너무 행복해" → ["sentiment", "text_gen"]
Input: "기분이 최악이야" → ["sentiment", "text_gen"]
Input: "이 글의 감정을 분석해줘" → ["sentiment", "text_gen"]"""

SYNTHESIS_SYSTEM = """한국어로 친근하고 자연스럽게 대화하세요.
사실 질문에는 정확한 정보를 간결하게 답하세요.
감성 분석 내용은 언급하지 마세요. 1~2문장만 답하세요.
한글로 작성하되, 고유명사(Washington D.C., Samsung 등)는 영어 그대로 써도 됩니다.
키릴 문자·한자·아랍어 등 라틴/한글 이외의 문자 사용 절대 금지."""

DOC_QA_SYSTEM = """당신은 문서 기반 AI 어시스턴트입니다.
[문서 발췌] 내용만을 근거로 [질문]에 답하세요.
문서에 없는 내용은 절대 추가하거나 유추하지 마세요. 반드시 2~3문장 이내로 간결하게 한국어로 답하세요."""
