'''
overlay_kb_json 추가해서 질문 수정 할 수 있게.
검색 순서 : overlay_kb_json(사용자가 수정한거) -> 캐시 -> llm

- 캐시나 overlay kb에서 유사성 좀 더 높일 수 있는 방법?
  (실제로 비슷한 질문인데 다르게 판단 하는 경우 있음)
'''
import json
import os
import math
from pathlib import Path
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import re

# 기존 import 아래에 추가
from sentence_transformers import CrossEncoder

# 환경 설정(client = OpenAI(...) 등) 아래에 추가
print("Reranker 모델을 로드하는 중입니다... (최초 실행 시 다운로드로 인해 시간이 걸릴 수 있습니다)")
reranker = CrossEncoder("Dongjin-kr/ko-reranker")

# =========================
# 환경 설정
# =========================
load_dotenv()
client = OpenAI(
    base_url="http://localhost:11434/v1",  # 로컬 Ollama 서버 주소
    api_key="ollama"  # Ollama는 키 검증을 하지 않으므로 임의의 값 입력
)

EMBEDDED_JSON = "guide_step5_embedded_chunk.json"
QA_CACHE_JSON = "qa_cache.json"
OVERLAY_KB_JSON = "overlay_kb.json"

TOP_K = 5
QA_SIM_THRESHOLD = 0.88
OVERLAY_SIM_THRESHOLD = 0.88


# =========================
# 유틸
# =========================
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def normalize_question(q: str):
    q = q.strip().lower()
    q = re.sub(r'[^\w\s가-힣]', '', q)
    q = re.sub(r'\s+', ' ', q)
    return q.strip()


def embed_query(text: str):
    res = client.embeddings.create(
        model="bge-m3",  # 👈 본인이 설치한 모델 이름으로 변경
        input=text
    )
    return res.data[0].embedding


def load_json(path, default):
    if not os.path.exists(path):
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def clear_json_file(path):
    save_json(path, [])
    return {
        "file": path,
        "cleared_at": datetime.now().isoformat()
    }

# =========================
# RAG
# =========================
def search_chunks(query: str):
    chunks = load_json(EMBEDDED_JSON, [])
    q_emb = embed_query(normalize_question(query))

    # [1단계] 임베딩(코사인 유사도) 기반 초벌 검색: 넉넉하게 상위 15개 추출
    scored = []
    for chunk in chunks:
        score = cosine_similarity(q_emb, chunk["embedding"])
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_15_chunks = scored[:15]

    if not top_15_chunks:
        return []

    # [2단계] Reranker 기반 정밀 재평가 (Cross-Encoder)
    # Reranker 입력 형태: [[질문, 문서내용1], [질문, 문서내용2], ...]
    cross_input = [[query, c[1]["content"]] for c in top_15_chunks]
    
    # 모델을 통해 연관성 점수 다시 계산 (점수가 높을수록 연관성이 높음)
    rerank_scores = reranker.predict(cross_input)
    
    # 새로운 Reranker 점수를 기준으로 문서와 매칭하여 재배열
    reranked = []
    for i, score in enumerate(rerank_scores):
        reranked.append((float(score), top_15_chunks[i][1]))
        
    reranked.sort(key=lambda x: x[0], reverse=True)
    
    # 최종적으로 가장 관련성 높은 핵심 문서 3개만 반환
    return reranked[:3]


def build_context(chunks):
    return "\n".join(
        f"{i}.\n제목: {c['title']}\n유형: {c['type']}\n내용:\n{c['content']}\n"
        for i, (_, c) in enumerate(chunks, 1)
    )


def generate_answer(query: str, context: str):
    res = client.chat.completions.create(
        model="llama3",  # 또는 설치하신 모델명 (예: llama3.1)
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 제공된 문서를 바탕으로 사용자의 질문에 답변하는 전문 AI 도우미입니다.\n"
                    "아래의 [지시사항]을 엄격하게 준수하여 답변하세요.\n\n"
                    "### 지시사항:\n"
                    "1. 반드시 제공된 [문서 컨텍스트]에 있는 정보만 사용하여 답변하세요.\n"
                    "2. 컨텍스트에 질문과 관련된 정보가 아예 없다면, 오직 \"제공된 문서에서는 해당 내용을 찾을 수 없습니다.\" 라고 **단 한 번만** 말하고 답변을 종료하세요. 절대 같은 문장을 반복하지 마세요.\n"
                    "3. 답변이 길어질 경우 글머리 기호(-, 1. 2.)나 굵은 글씨 등 마크다운(Markdown)을 활용하여 가독성 있게 정리하세요.\n"
                    "4. 모든 대답은 반드시 한국어(Korean)로만 작성하세요."
                )
            },
            {
                "role": "user",
                "content": f"[문서 컨텍스트]\n{context}\n\n[질문]\n{query}"
            }
        ],
        temperature=0.1 # 0은 너무 뻣뻣해질 수 있어 0.1 정도로 살짝 올리는 것도 좋습니다.
    )
    return res.choices[0].message.content


# =========================
# QA Cache
# =========================
def search_qa_cache(query: str):
    cache = load_json(QA_CACHE_JSON, [])
    if not cache:
        return None, 0.0

    q_emb = embed_query(normalize_question(query))
    best, best_score = None, 0.0

    for item in cache:
        score = cosine_similarity(q_emb, item["question_embedding"])
        if score > best_score:
            best, best_score = item, score

    if best_score >= QA_SIM_THRESHOLD:
        return best, best_score
    return None, best_score


def add_qa_cache(query: str, answer: str):
    cache = load_json(QA_CACHE_JSON, [])
    cache.append({
        "created_at": datetime.now().isoformat(),
        "question": query,
        "answer": answer,
        "question_embedding": embed_query(normalize_question(query))
    })
    save_json(QA_CACHE_JSON, cache)


def purge_qa_cache(question: str):
    cache = load_json(QA_CACHE_JSON, [])
    q_emb = embed_query(normalize_question(question))

    new_cache, removed = [], []
    for item in cache:
        if cosine_similarity(q_emb, item["question_embedding"]) >= QA_SIM_THRESHOLD:
            removed.append(item["question"])
        else:
            new_cache.append(item)

    save_json(QA_CACHE_JSON, new_cache)
    return removed


# =========================
# Overlay KB
# =========================
def purge_overlay_kb(question: str):
    overlay = load_json(OVERLAY_KB_JSON, [])
    if not overlay:
        return overlay, None

    q_emb = embed_query(normalize_question(question))

    new_overlay = []
    removed_item = None

    for item in overlay:
        score = cosine_similarity(q_emb, item["question_embedding"])
        if score >= OVERLAY_SIM_THRESHOLD:
            removed_item = item
        else:
            new_overlay.append(item)

    return new_overlay, removed_item

def search_overlay_kb(query: str):
    overlay = load_json(OVERLAY_KB_JSON, [])
    if not overlay:
        return None, 0.0

    q_emb = embed_query(normalize_question(query))
    best, best_score = None, 0.0

    for item in overlay:
        emb = item.get("question_embedding")
        if not emb:
            emb = embed_query(normalize_question(item["question"]))
            item["question_embedding"] = emb

        score = cosine_similarity(q_emb, emb)
        if score > best_score:
            best, best_score = item, score

    if best_score >= OVERLAY_SIM_THRESHOLD:
        save_json(OVERLAY_KB_JSON, overlay)
        return best, best_score

    return None, best_score


# =========================
# Trace
# =========================
def make_trace(question, source, score, rag_titles):
    return {
        "question": question,
        "source": source,
        "score": round(score, 4),
        "rag_chunks": rag_titles
    }


# =========================
# Gradio Logic
# =========================
def chat(query, chat_history, trace_history):
    # 1️⃣ Overlay 우선
    overlay_item, overlay_score = search_overlay_kb(query)
    if overlay_item:
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": overlay_item["answer"]})
        trace_history.append(make_trace(query, "overlay", overlay_score, []))
        return chat_history, trace_history, ""

    # 2️⃣ QA Cache
    cache_item, cache_score = search_qa_cache(query)
    if cache_item:
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": cache_item["answer"]})
        trace_history.append(make_trace(query, "qa_cache", cache_score, []))
        return chat_history, trace_history, ""

    # 3️⃣ RAG + LLM
    chunks = search_chunks(query)
    context = build_context(chunks)
    answer = generate_answer(query, context)

    # 4️⃣ ⭐ QA Cache 저장 여부 판단 : llm을 사용한 경우, overlay에 비슷한 다변이 있는지 검사 후 캐시에 저장
    overlay_item, overlay_score = search_overlay_kb(query)
    if overlay_item is None:
        add_qa_cache(query, answer)

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})
    trace_history.append(
        make_trace(query, "llm", 0.0, [c[1]["title"] for c in chunks])
    )

    return chat_history, trace_history, ""



def select_chat(evt: gr.SelectData, chat_history):
    idx = evt.index
    item = chat_history[idx]

    if item["role"] == "assistant":
        question = chat_history[idx - 1]["content"]
        answer = item["content"]

    elif item["role"] == "user":
        question = item["content"]
        # 다음이 assistant인지 체크
        if idx + 1 < len(chat_history) and chat_history[idx + 1]["role"] == "assistant":
            answer = chat_history[idx + 1]["content"]
        else:
            return "", "", gr.update()

    else:
        return "", "", gr.update()

    return (
        question,
        answer,
        answer,
        gr.update(value="수정")  # ⭐ 라디오 자동 변경
    )


def strip_embedding(item: dict):
    """UI 표시용: question_embedding 제거"""
    return {
        k: v for k, v in item.items()
        if k != "question_embedding"
    }

def save_overlay(user, mode, q, edited_a, original_a):
    overlay = load_json(OVERLAY_KB_JSON, [])

    new_item = {
        "user": user,
        "created_at": datetime.now().isoformat(),
        "mode": mode,  # ✅ 수정 / 새로 생성
        "question": q,
        "original_answer": None if mode == "새로 생성" else original_a,
        "answer": edited_a,
        "question_embedding": embed_query(normalize_question(q))
    }

    overlay.append(new_item)
    save_json(OVERLAY_KB_JSON, overlay)

    purge_qa_cache(q)


    return strip_embedding(new_item)



# =========================
# Gradio UI
# =========================

def on_overlay_mode_change(mode):
    if mode == "수정":
        return (
            gr.update(label="질문"),
            gr.update(label="수정 답변")
        )
    else:  # 새로 생성
        return (
            gr.update(label="새로운 정보", value=""),
            gr.update(label="답변", value="")
        )

with gr.Blocks(title="Test Bot") as demo:
    # gr.Markdown("## 🧪 Overlay KB + QA Cache + RAG 테스트")
    gr.Markdown("## 🧪 MLOps 챗봇 - 궁금한것을 물어보세요")

    chat_state = gr.State([])
    trace_state = gr.State([])
    original_answer_state = gr.State("")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="💬 Chat")
            query_box = gr.Textbox(label="질문 입력")
            send_btn = gr.Button("전송")


    gr.Markdown("## ✍️ Overlay KB - 대화를 클릭해 수정하거나 답변을 새로 생성하세요.")

    user_box = gr.Textbox(label="수정자", value="jeeeun")
    overlay_mode = gr.Radio(
        ["수정", "새로 생성"],
        value="수정",
        label="Overlay 유형"
    )
    edit_q = gr.Textbox(label="질문")
    edit_a = gr.Textbox(label="수정 답변", lines=6)

    save_btn = gr.Button("Overlay KB 저장")
    overlay_view = gr.JSON(label="📦 Overlay 결과")

    with gr.Column(scale=2):
        with gr.Row():
            clear_overlay_btn = gr.Button("🧹 Overlay KB 초기화")
            clear_cache_btn = gr.Button("🧹 QA Cache 초기화")
        trace_view = gr.JSON(label="🔍 Trace")

    send_btn.click(chat, [query_box, chat_state, trace_state],
                   [chatbot, trace_view, query_box])

    query_box.submit(chat, [query_box, chat_state, trace_state],
                     [chatbot, trace_view, query_box])

    chatbot.select(
        select_chat,
        [chat_state],
        [edit_q, edit_a, original_answer_state, overlay_mode]
    )

    save_btn.click(
        save_overlay,
        [user_box, overlay_mode, edit_q, edit_a, original_answer_state],
        overlay_view
    )

    clear_overlay_btn.click(
        lambda: clear_json_file(OVERLAY_KB_JSON),
        outputs=trace_view
    )

    clear_cache_btn.click(
        lambda: clear_json_file(QA_CACHE_JSON),
        outputs=trace_view
    )

    overlay_mode.change(
        on_overlay_mode_change,
        inputs=overlay_mode,
        outputs=[edit_q, edit_a]
    )

demo.launch(server_port=7860)
# demo.launch()
