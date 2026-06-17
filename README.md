# tomAI SLM Pipeline Server

다중 NLP 태스크를 처리하는 SLM 추론 서버 (FastAPI + MLX)

## 모델 구성

- 텍스트 생성 / 감성 분석 / 번역 / QnA: Gemma-2-2B-it (4bit, MLX)
- 개체명 인식(NER): KoELECTRA-base-v3-naver-ner

모델은 최초 실행 시 HuggingFace Hub에서 자동 다운로드되며, 별도의 로컬 가중치 파일을 저장소에 포함하지 않습니다.

## 동작 방식

1. 사용자 입력을 분석해 적용할 파이프라인을 동적으로 라우팅 (`detect_pipelines`)
2. text_gen / sentiment / translation / ner / qna 중 해당 파이프라인 순차 실행
3. 결과를 통합해 SSE로 스트리밍 응답

## 실행

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python main.py
```

## 서비스 연동

별도의 공개 엔드포인트는 없으며, 내부 전용(localhost) 서버입니다.
[tomAI 웹서버](https://github.com/orioti-anne/tomai)가 [tomai.orioti.com/slm](https://tomai.orioti.com/slm)에서
이 서버로 내부 프록시 요청(SSE)을 전달하는 구조입니다.
