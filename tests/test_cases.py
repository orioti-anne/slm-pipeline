"""다양한 케이스 테스트."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pipelines.sentiment as sentiment
import pipelines.ner as ner
import pipelines.translation as translation
import pipelines.qna as qna
import pipelines.text_gen as text_gen

PASS = "✓"
FAIL = "✗"

def section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)

def case(label, result, check_fn=None):
    ok = check_fn(result) if check_fn else True
    mark = PASS if ok else FAIL
    print(f"  [{mark}] {label}")
    print(f"       → {result}")
    return ok


# ── Sentiment ──────────────────────────────────────────
section("Sentiment")

case("긍정 - 명백한 기쁨",
    sentiment.run("오늘 정말 행복하고 즐거운 하루였어요!"),
    lambda r: r["label"] == "긍정")

case("부정 - 불만 표현",
    sentiment.run("서비스가 너무 별로고 완전 실망스러웠어요."),
    lambda r: r["label"] == "부정")

case("중립 - 사실 진술",
    sentiment.run("내일 회의는 오후 3시에 시작합니다."),
    lambda r: r["label"] == "중립")

case("긍정 - 금융 뉴스",
    sentiment.run("삼성전자 주가가 사상 최고치를 기록했습니다."),
    lambda r: "label" in r)

case("부정 - 짧은 표현",
    sentiment.run("최악이야."),
    lambda r: r["label"] == "부정")


# ── NER ────────────────────────────────────────────────
section("NER")

case("사람+장소+조직+날짜",
    ner.run("홍길동이 2024년 3월 서울에서 삼성전자를 방문했다."),
    lambda r: len(r["entities"]) > 0)

case("영문 혼재",
    ner.run("이순신 장군은 조선시대 명장으로 한산도 대첩을 승리로 이끌었다."),
    lambda r: len(r["entities"]) > 0)

case("여러 인물",
    ner.run("박지성과 손흥민은 한국을 대표하는 축구 선수다."),
    lambda r: len(r["entities"]) >= 2)

case("날짜+금액",
    ner.run("2023년 12월에 카카오가 1조원 규모의 투자를 유치했다."),
    lambda r: len(r["entities"]) > 0)

case("엔티티 없는 문장",
    ner.run("오늘 날씨가 매우 맑고 따뜻하다."),
    lambda r: "entities" in r)


# ── Translation ────────────────────────────────────────
section("Translation")

case("KO→EN 인사",
    translation.run("안녕하세요, 만나서 반갑습니다.", direction="ko_en"),
    lambda r: len(r["translated"]) > 0)

case("KO→EN 긴 문장",
    translation.run("인공지능 기술의 발전으로 많은 산업 분야에서 혁신이 일어나고 있습니다.", direction="ko_en"),
    lambda r: len(r["translated"]) > 0)

case("EN→KO 인사",
    translation.run("Hello, nice to meet you.", direction="en_ko"),
    lambda r: len(r["translated"]) > 0)

case("EN→KO 긴 문장",
    translation.run("Artificial intelligence is transforming many industries around the world.", direction="en_ko"),
    lambda r: len(r["translated"]) > 0)

case("KO→EN 숫자/날짜 포함",
    translation.run("2024년 3월 15일에 서울에서 열린 행사에 1만 명이 참가했습니다.", direction="ko_en"),
    lambda r: len(r["translated"]) > 0)


# ── QnA ────────────────────────────────────────────────
section("QnA")

ctx1 = "대한민국의 수도는 서울이며, 인구는 약 950만 명이다."
case("수도 질문",
    qna.run("대한민국의 수도는 어디인가요?", context=ctx1),
    lambda r: "서울" in r["answer"])

ctx2 = "세종대왕은 1397년에 태어나 1450년에 사망했으며, 훈민정음을 창제한 조선의 왕이다."
case("인물 사망 연도",
    qna.run("세종대왕은 언제 사망했나요?", context=ctx2),
    lambda r: "1450" in r["answer"])

case("인물 업적",
    qna.run("세종대왕의 업적은 무엇인가요?", context=ctx2),
    lambda r: len(r["answer"]) > 0)

ctx3 = "삼성전자는 1969년에 설립되었으며 본사는 경기도 수원에 위치한다. 현재 직원 수는 약 27만 명이다."
case("회사 설립 연도",
    qna.run("삼성전자는 언제 설립되었나요?", context=ctx3),
    lambda r: "1969" in r["answer"])

case("context 없을 때",
    qna.run("질문", context=""),
    lambda r: r.get("error") is not None)


# ── Router ─────────────────────────────────────────────
section("Router (detect_pipelines)")

cases = [
    ("순수 대화", "안녕하세요! 오늘 기분 어때요?", ["text_gen"]),
    ("번역 요청", "이 문장을 영어로 번역해줘: 안녕하세요", ["translation"]),
    ("감성 분석", "이 리뷰의 감정을 분석해줘: 정말 최고예요!", ["sentiment", "text_gen"]),
    ("복합 인텐트", "오늘 기분이 너무 안좋아, 서울 빵집 추천해줘", None),
    ("영어 번역", "Hello, translate this to Korean", ["translation"]),
    ("NER 요청", "이 문장에서 사람 이름을 찾아줘: 홍길동이 서울에 갔다", None),
]

for label, text, expected in cases:
    result = text_gen.detect_pipelines(text)
    if expected:
        ok = set(expected).issubset(set(result))
    else:
        ok = len(result) > 0
    mark = PASS if ok else FAIL
    print(f"  [{mark}] {label}: '{text[:30]}...' " if len(text) > 30 else f"  [{mark}] {label}: '{text}'")
    print(f"       → {result}")


# ── Text Gen ───────────────────────────────────────────
section("Text Gen")

prompts = [
    "파이썬과 자바스크립트의 차이점을 간단히 설명해줘.",
    "오늘 점심 메뉴 추천해줘.",
    "머신러닝이 뭔지 초등학생도 이해할 수 있게 설명해줘.",
]

for prompt in prompts:
    result = text_gen.run(prompt)
    ok = len(result) > 20
    mark = PASS if ok else FAIL
    print(f"  [{mark}] '{prompt[:40]}...' " if len(prompt) > 40 else f"  [{mark}] '{prompt}'")
    print(f"       → {result[:120]}{'...' if len(result) > 120 else ''}")

print(f"\n{'='*50}")
print("  테스트 완료")
print('='*50)
