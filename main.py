import re
import json
import gc
import os
import time
import threading
import secrets
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

import pipelines.text_gen as text_gen
import pipelines.translation as translation
import pipelines.sentiment as sentiment
import pipelines.ner as ner
import pipelines.qna as qna
from config import PORT, HOST, DOC_QA_SYSTEM

IDLE_TIMEOUT = 600  # 10분

_last_activity = time.time()
_MODEL_KEYS = ["gen_model", "gen_tokenizer", "ner"]


def _unload_models():
    from core import model_manager
    for k in _MODEL_KEYS:
        model_manager.clear(k)
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def _idle_watcher():
    global _last_activity
    while True:
        time.sleep(60)
        if time.time() - _last_activity > IDLE_TIMEOUT:
            _unload_models()
            _last_activity = time.time()


threading.Thread(target=_idle_watcher, daemon=True).start()


# 업로드된 문서 context를 메모리에 보관 (doc_id → text)
_doc_store: dict[str, str] = {}


def _clean_context(text: str) -> str:
    """PDF 아티팩트 제거: '# #' 헤더, 단독 숫자 줄(페이지 번호), 과도한 공백."""
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _find_relevant_chunk(context: str, question: str, window: int = 1500) -> str:
    """질문 키워드 기반으로 문서에서 가장 관련성 높은 구간을 반환."""
    if len(context) <= window:
        return context
    q_words = set(w.lower() for w in re.findall(r'[a-zA-Z가-힣]{2,}', question))
    if not q_words:
        return context[:window]
    ctx_lower = context.lower()
    best_start, best_score = 0, -1
    for start in range(0, len(context) - window + 1, 200):
        score = sum(ctx_lower[start:start + window].count(w) for w in q_words)
        if score > best_score:
            best_score, best_start = score, start
    return context[best_start:best_start + window]


PIPELINE_MAP = {
    "translation": translation,
    "sentiment": sentiment,
    "ner": ner,
    "qna": qna,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(text_gen._load)
    yield


app = FastAPI(title="SLM Pipeline", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request 모델 ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    text: str
    context: str = ""

class PipelineRequest(BaseModel):
    text: str
    context: str = ""
    direction: str = "ko_en"


class FetchDocRequest(BaseModel):
    url: str

# ── UI ───────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>범용 Chat — SLM Pipeline</title>
<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 min-h-screen">

<header class="sticky top-0 z-50 bg-white border-b border-slate-200 shadow-sm">
  <div class="max-w-3xl mx-auto px-4 h-14 flex items-center gap-3">
    <div class="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white text-xs font-bold">SLM</div>
    <span class="font-bold text-slate-800">범용 Chat</span>
    <span class="ml-1 text-xs text-slate-400">5-Core Pipeline</span>
    <div id="status-dot" class="ml-auto w-2 h-2 rounded-full bg-slate-300"></div>
    <span id="status-text" class="text-xs text-slate-400">대기 중</span>
  </div>
</header>

<main class="max-w-3xl mx-auto px-4 py-6 space-y-4">

  <!-- 입력 카드 -->
  <div class="bg-white rounded-2xl border border-slate-200 p-4 shadow-sm">
    <div class="space-y-3">
      <textarea id="input" rows="3"
        class="w-full resize-none border border-slate-200 rounded-xl p-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300 placeholder-slate-400"
        placeholder="텍스트를 입력하세요... (Shift+Enter 줄바꿈, Enter 전송)"></textarea>
      <div class="flex gap-2">
        <input id="context" type="text"
          class="flex-1 border border-slate-200 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-200 placeholder-slate-400"
          placeholder="QnA context (선택 — 문서나 지문을 붙여넣으세요)">
        <button onclick="sendMessage()" id="send-btn"
          class="px-5 py-2 bg-indigo-600 text-white text-sm font-semibold rounded-xl hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors">
          전송
        </button>
      </div>
    </div>
  </div>

  <!-- 출력 영역 -->
  <div id="output" class="hidden space-y-3">

    <!-- 파이프라인 배지 -->
    <div id="pipeline-badges" class="flex flex-wrap gap-2 px-1"></div>

    <!-- 파이프라인 결과 카드들 -->
    <div id="pipeline-results" class="space-y-2"></div>

    <!-- 최종 답변 카드 -->
    <div id="textgen-card" class="hidden bg-white rounded-2xl border border-indigo-100 p-4 shadow-sm">
      <div class="flex items-center gap-2 mb-3">
        <span class="text-sm">✨</span>
        <span class="text-xs font-semibold text-slate-500">최종 답변</span>
        <span id="textgen-status" class="ml-auto text-xs text-indigo-400"></span>
      </div>
      <div id="textgen-output" class="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap"></div>
    </div>
  </div>

</main>

<script>
const BADGE_STYLE = {
  sentiment:   'bg-rose-100 text-rose-700 border border-rose-200',
  ner:         'bg-amber-100 text-amber-700 border border-amber-200',
  translation: 'bg-sky-100 text-sky-700 border border-sky-200',
  qna:         'bg-emerald-100 text-emerald-700 border border-emerald-200',
  text_gen:    'bg-indigo-100 text-indigo-700 border border-indigo-200',
};
const BADGE_LABEL = {
  sentiment: '😊 감성 분석', ner: '🏷️ 개체명', translation: '🌐 번역',
  qna: '❓ QnA', text_gen: '✨ 텍스트 생성',
};

let currentES = null;

function sendMessage() {
  const text = document.getElementById('input').value.trim();
  if (!text) return;
  if (currentES) { currentES.close(); }

  const context = document.getElementById('context').value.trim();

  // UI 초기화
  document.getElementById('output').classList.remove('hidden');
  document.getElementById('pipeline-badges').innerHTML = '';
  document.getElementById('pipeline-results').innerHTML = '';
  document.getElementById('textgen-card').classList.add('hidden');
  document.getElementById('textgen-output').textContent = '';
  document.getElementById('textgen-status').textContent = '';
  setStatus('처리 중...', 'indigo');
  setSendBtn(false);

  const params = new URLSearchParams({ text });
  if (context) params.set('context', context);

  const es = new EventSource('/chat/stream?' + params);
  currentES = es;

  es.onmessage = (e) => {
    if (e.data === '[DONE]') {
      es.close(); currentES = null;
      setSendBtn(true);
      setStatus('완료', 'green');
      document.getElementById('textgen-status').textContent = '완료';
      return;
    }
    const msg = JSON.parse(e.data);
    if (msg.type === 'pipelines') {
      renderBadges(msg.value);
    } else if (msg.type === 'status') {
      setStatus(msg.value, 'indigo');
      const pipe = msg.value.split(' ')[0];
      activateBadge(pipe);
    } else if (msg.type === 'pipeline_result') {
      renderResult(msg.pipeline, msg.value);
    } else if (msg.type === 'token') {
      const card = document.getElementById('textgen-card');
      card.classList.remove('hidden');
      document.getElementById('textgen-status').textContent = '생성 중...';
      document.getElementById('textgen-output').textContent += msg.value;
    }
  };

  es.onerror = () => {
    es.close(); currentES = null;
    setSendBtn(true);
    setStatus('오류 발생', 'red');
  };
}

function setSendBtn(enabled) {
  const btn = document.getElementById('send-btn');
  btn.disabled = !enabled;
  btn.textContent = enabled ? '전송' : '처리 중...';
}

function setStatus(msg, color) {
  const dot = document.getElementById('status-dot');
  const txt = document.getElementById('status-text');
  const colors = { indigo: 'bg-indigo-400', green: 'bg-green-400', red: 'bg-red-400', slate: 'bg-slate-300' };
  dot.className = 'ml-auto w-2 h-2 rounded-full ' + (colors[color] || colors.slate);
  txt.textContent = msg;
}

function renderBadges(pipelines) {
  const c = document.getElementById('pipeline-badges');
  pipelines.forEach(p => {
    const s = document.createElement('span');
    s.id = 'badge-' + p;
    s.className = 'inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold opacity-40 transition-opacity ' + (BADGE_STYLE[p] || 'bg-slate-100 text-slate-600');
    s.textContent = BADGE_LABEL[p] || p;
    c.appendChild(s);
  });
}

function activateBadge(pipe) {
  const b = document.getElementById('badge-' + pipe);
  if (b) b.classList.replace('opacity-40', 'opacity-100');
}

function renderResult(pipeline, value) {
  activateBadge(pipeline);
  const c = document.getElementById('pipeline-results');
  const card = document.createElement('div');
  card.className = 'bg-white rounded-xl border border-slate-100 p-3 shadow-sm';

  let inner = '';
  if (pipeline === 'sentiment') {
    const emoji = value.label === '긍정' ? '😊' : value.label === '부정' ? '😞' : '😐';
    inner = `<div class="flex items-center gap-3">
      <span class="text-2xl">${emoji}</span>
      <div>
        <div class="font-semibold text-slate-700">${value.label}</div>
        <div class="text-xs text-slate-400">신뢰도 ${(value.score * 100).toFixed(1)}%</div>
      </div>
    </div>`;
  } else if (pipeline === 'ner') {
    const tags = value.entities.length
      ? value.entities.map(e =>
          `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-amber-50 border border-amber-200 text-xs">
            <span class="font-semibold text-amber-800">${e.text}</span>
            <span class="text-amber-400">${e.type}</span>
          </span>`).join(' ')
      : '<span class="text-xs text-slate-400">감지된 개체 없음</span>';
    inner = `<div class="flex flex-wrap gap-1.5">${tags}</div>`;
  } else if (pipeline === 'translation') {
    const arrow = value.direction === 'ko_en' ? '한 → 영' : '영 → 한';
    inner = `<div class="text-sm text-slate-700">
      <span class="text-xs text-sky-400 font-semibold mr-2">${arrow}</span>${value.translated}
    </div>`;
  } else if (pipeline === 'qna') {
    inner = `<div class="text-sm font-semibold text-slate-700">${value.answer || '<span class="text-slate-400 font-normal">답변 없음</span>'}</div>`;
  } else {
    inner = `<pre class="text-xs text-slate-500 overflow-auto">${JSON.stringify(value, null, 2)}</pre>`;
  }

  const label = BADGE_LABEL[pipeline] || pipeline;
  card.innerHTML = `
    <div class="flex items-center gap-1.5 mb-2">
      <span class="text-xs font-semibold text-slate-400">${label}</span>
    </div>
    ${inner}`;
  c.appendChild(card);
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
});
</script>
</body>
</html>"""


@app.post("/upload-doc")
async def upload_doc(file: UploadFile = File(...)):
    """PDF / DOCX / TXT 문서에서 텍스트를 추출해 반환."""
    data = await file.read()
    name = file.filename or ""
    ext = name.rsplit(".", 1)[-1].lower()

    try:
        if ext == "pdf":
            import pdfplumber, io
            text_parts = []
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                pages = len(pdf.pages)
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_parts.append(t)
            text = "\n\n".join(text_parts)

        elif ext in ("docx", "doc"):
            import docx, io
            doc = docx.Document(io.BytesIO(data))
            pages = None
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        elif ext == "txt":
            import chardet
            enc = chardet.detect(data)["encoding"] or "utf-8"
            text = data.decode(enc)
            pages = None

        else:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식: .{ext} (PDF, DOCX, TXT만 가능)")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"텍스트 추출 실패: {str(e)}")

    text = _clean_context(text)
    if not text:
        raise HTTPException(status_code=422, detail="문서에서 텍스트를 추출할 수 없습니다. 이미지 기반 PDF일 수 있습니다.")

    doc_id = secrets.token_urlsafe(8)
    _doc_store[doc_id] = text

    return {
        "filename": name,
        "ext": ext,
        "pages": pages,
        "chars": len(text),
        "doc_id": doc_id,
    }


@app.delete("/doc/{doc_id}")
async def delete_doc(doc_id: str):
    _doc_store.pop(doc_id, None)
    return {"deleted": doc_id}

@app.post("/fetch-doc")
async def fetch_doc(req: FetchDocRequest):
    """URL에서 텍스트 추출 (Google Docs/Drive, 일반 웹페이지) -> doc_id 반환."""
    import re as _re2
    import requests as _req
    from urllib.parse import urlparse

    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="http:// 또는 https://로 시작하는 URL이어야 합니다.")

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

    try:
        gdoc_m = _re2.search(r"docs\.google\.com/document/d/([a-zA-Z0-9_-]+)", url)
        gdrive_m = _re2.search(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", url)
        gdrive_open_m = _re2.search(r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)", url)

        if gdoc_m:
            gid = gdoc_m.group(1)
            export_url = f"https://docs.google.com/document/d/{gid}/export?format=txt"
            resp = await asyncio.to_thread(_req.get, export_url, headers=headers, timeout=15, allow_redirects=True)
            if "accounts.google.com" in resp.url or "ServiceLogin" in resp.text[:500]:
                raise HTTPException(status_code=403, detail="비공개 문서입니다. Google Docs 공유 설정을 '링크가 있는 모든 사용자'로 변경해주세요.")
            resp.raise_for_status()
            text = resp.text
            name = "Google Docs"

        elif gdrive_m or gdrive_open_m:
            fid = (gdrive_m or gdrive_open_m).group(1)
            download_url = f"https://drive.google.com/uc?export=download&id={fid}"
            resp = await asyncio.to_thread(_req.get, download_url, headers=headers, timeout=15, allow_redirects=True)
            if "accounts.google.com" in resp.url:
                raise HTTPException(status_code=403, detail="비공개 파일입니다. Google Drive 공유 설정을 '링크가 있는 모든 사용자'로 변경해주세요.")
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "text" in ct:
                text = resp.text
            elif "application/pdf" in ct:
                import pdfplumber, io
                parts = []
                with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            parts.append(t)
                text = "\n\n".join(parts)
            else:
                raise HTTPException(status_code=422, detail=f"지원하지 않는 파일 형식입니다. ({ct})")
            name = "Google Drive"

        else:
            resp = await asyncio.to_thread(_req.get, url, headers=headers, timeout=15, allow_redirects=True)
            resp.raise_for_status()
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
                tag.decompose()
            main_el = (
                soup.find("main") or soup.find("article") or
                soup.find(id="content") or soup.find(id="main") or soup.find("body")
            )
            text = main_el.get_text(separator="\n", strip=True) if main_el else soup.get_text(separator="\n", strip=True)
            name = urlparse(url).netloc

    except HTTPException:
        raise
    except _req.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="URL 응답 시간 초과 (15초)")
    except _req.exceptions.ConnectionError:
        raise HTTPException(status_code=502, detail="URL에 연결할 수 없습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"불러오기 실패: {str(e)}")

    text = _clean_context(text)
    if not text or len(text) < 50:
        raise HTTPException(status_code=422, detail="URL에서 충분한 텍스트를 추출할 수 없습니다.")

    doc_id = secrets.token_urlsafe(8)
    _doc_store[doc_id] = text
    return {"filename": name, "chars": len(text), "doc_id": doc_id}



@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_HTML)


# ── 엔드포인트 ───────────────────────────────────────────────

@app.get("/health")
async def health():
    from core import model_manager
    idle_sec = int(time.time() - _last_activity)
    loaded = [k for k in _MODEL_KEYS if model_manager.loaded(k)]
    return {"status": "ok", "version": "1.0", "idle_sec": idle_sec, "loaded": loaded}


@app.post("/chat")
async def chat(req: ChatRequest):
    """REST: 전체 처리 완료 후 한 번에 반환."""
    pipelines_needed = await asyncio.to_thread(text_gen.detect_pipelines, req.text)

    if pipelines_needed == ["text_gen"]:
        reply = await asyncio.to_thread(text_gen.run, req.text)
        return {"reply": reply, "pipelines": pipelines_needed}

    analysis_tasks = []
    for p in pipelines_needed:
        if p == "text_gen":
            continue
        if p == "qna":
            analysis_tasks.append((p, asyncio.to_thread(qna.run, req.text, req.context)))
        else:
            analysis_tasks.append((p, asyncio.to_thread(PIPELINE_MAP[p].run, req.text)))

    results = {}
    for name, coro in analysis_tasks:
        results[name] = await coro

    if "text_gen" in pipelines_needed:
        synthesis_input = (
            f"사용자 입력: {req.text}\n"
            f"분석 결과: {json.dumps(results, ensure_ascii=False)}"
        )
        reply = await asyncio.to_thread(text_gen.run, synthesis_input)
    else:
        reply = json.dumps(results, ensure_ascii=False)

    return {"reply": reply, "pipelines": pipelines_needed, "details": results}


@app.get("/chat/stream")
async def chat_stream(text: str, context: str = "", doc_id: str = ""):
    """SSE: 파이프라인 상태 + 토큰을 실시간으로 스트리밍."""

    # doc_id로 context 복원. context="undefined"는 브라우저 캐시 버그로 무시
    if doc_id:
        resolved_context = _doc_store.get(doc_id, "")
    elif context and context != "undefined":
        resolved_context = context
    else:
        resolved_context = ""

    def generate():
        global _last_activity
        _last_activity = time.time()
        if resolved_context:
            # 문서 모드: KoELECTRA로 문서에서 직접 스팬 추출
            pipelines_needed = ["qna"]
        else:
            pipelines_needed = text_gen.detect_pipelines(text)
            # 문서 없을 때 qna는 의미 없으므로 제거
            pipelines_needed = [p for p in pipelines_needed if p != "qna"]
            # 번역이 포함되면 text_gen 불필요 (번역 결과가 곧 답변)
            if "translation" in pipelines_needed:
                pipelines_needed = ["translation"]
            elif "text_gen" not in pipelines_needed:
                pipelines_needed.append("text_gen")

        yield f"data: {json.dumps({'type': 'pipelines', 'value': pipelines_needed}, ensure_ascii=False)}\n\n"

        results = {}
        for p in pipelines_needed:
            if p == "text_gen":
                continue
            yield f"data: {json.dumps({'type': 'status', 'value': f'{p} 처리 중...'})}\n\n"
            if p == "qna":
                result = qna.run(text, resolved_context)
            elif p == "translation":
                en_ko_kw = ["한국어로", "한글로", "국문으로", "한국말로"]
                direction = "en_ko" if any(k in text for k in en_ko_kw) else "ko_en"
                result = PIPELINE_MAP[p].run(text, direction)
            else:
                result = PIPELINE_MAP[p].run(text)
            results[p] = result
            yield f"data: {json.dumps({'type': 'pipeline_result', 'pipeline': p, 'value': result}, ensure_ascii=False)}\n\n"

        if resolved_context:
            # 문서 모드: QnA 카드가 답변. 못 찾으면 메시지만 출력
            if not results.get("qna", {}).get("answer"):
                yield f"data: {json.dumps({'type': 'token', 'value': '문서에서 관련 내용을 찾지 못했습니다.'}, ensure_ascii=False)}\n\n"
        elif "text_gen" in pipelines_needed:
            yield f"data: {json.dumps({'type': 'status', 'value': '답변 생성 중...'})}\n\n"
            synthesis_input = text
            for chunk in text_gen.stream(synthesis_input):
                yield f"data: {json.dumps({'type': 'token', 'value': chunk}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/pipeline/{name}")
async def run_pipeline(name: str, req: PipelineRequest):
    """개별 파이프라인 직접 호출."""
    if name == "text_gen":
        result = await asyncio.to_thread(text_gen.run, req.text)
        return {"pipeline": name, "result": result}
    if name == "translation":
        result = await asyncio.to_thread(translation.run, req.text, req.direction)
        return {"pipeline": name, "result": result}
    if name == "qna":
        result = await asyncio.to_thread(qna.run, req.text, req.context)
        return {"pipeline": name, "result": result}
    if name in PIPELINE_MAP:
        result = await asyncio.to_thread(PIPELINE_MAP[name].run, req.text)
        return {"pipeline": name, "result": result}
    raise HTTPException(status_code=404, detail=f"Unknown pipeline: {name}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
