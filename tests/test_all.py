"""각 파이프라인 단독 테스트 스크립트."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pipelines.sentiment as sentiment
import pipelines.ner as ner
import pipelines.translation as translation
import pipelines.qna as qna
import pipelines.text_gen as text_gen


def test_sentiment():
    print("\n[Sentiment]")
    result = sentiment.run("오늘 기분이 너무 좋아요!")
    print(result)
    assert "label" in result

def test_ner():
    print("\n[NER]")
    result = ner.run("홍길동이 2024년 3월 서울에서 삼성전자를 방문했다.")
    print(result)
    assert "entities" in result

def test_translation_ko_en():
    print("\n[Translation KO→EN]")
    result = translation.run("안녕하세요, 만나서 반갑습니다.", direction="ko_en")
    print(result)
    assert "translated" in result

def test_translation_en_ko():
    print("\n[Translation EN→KO]")
    result = translation.run("Hello, nice to meet you.", direction="en_ko")
    print(result)
    assert "translated" in result

def test_qna():
    print("\n[QnA]")
    context = "대한민국의 수도는 서울이며, 인구는 약 950만 명이다."
    result = qna.run("대한민국의 수도는 어디인가요?", context=context)
    print(result)
    assert "answer" in result

def test_router():
    print("\n[Router]")
    cases = [
        "오늘 기분이 너무 안좋아, 서울 빵집 추천해줘",
        "이 문장을 영어로 번역해줘: 안녕하세요",
        "안녕하세요!",
    ]
    for text in cases:
        result = text_gen.detect_pipelines(text)
        print(f"  '{text}' → {result}")

def test_text_gen():
    print("\n[Text Gen]")
    result = text_gen.run("토마토 재배에서 가장 중요한 점은 무엇인가요?")
    print(result[:200])
    assert len(result) > 0


if __name__ == "__main__":
    # 가벼운 파이프라인 먼저 (transformers)
    test_sentiment()
    test_ner()
    test_translation_ko_en()
    test_translation_en_ko()
    test_qna()
    # MLX 모델 (마지막 — 로드 시간 있음)
    test_router()
    test_text_gen()
    print("\n✓ 모든 테스트 통과")
