"""
[Step 2] 실험·실습실 안전관리 RAG 챗봇 (Streamlit UI - 3가지 모드 선택 통합 버전)
전주대학교 AI융합프로젝트 (김영수 교수님) 강의안 전 과정 통합 구현:
1. ⚡ 오프라인 RAG 엔진 버전: API 키 및 외부 호출 없이 로컬 BM25/MMR 검색 + 규칙 기반 구조화 요약으로 0.1초 즉시 답변
2. 🦙 무료 오픈소스 LLM 버전 (강의안 AdvancedRAG p.12~19): Ollama 로컬 오픈소스 모델(qwen2.5:3b, exaone3.5:2.4b, gemma2:2b, llama3.2:3b) 연동 (API 키 불필요 · 0원)
3. ☁️ 클라우드 API 버전: 초경량 모델(gemini-2.5-flash-lite / gemini-2.0-flash-lite) + 동일 질문 캐싱으로 최소 크레딧 소모
+ 🌐 영어 원문 PDF 업로드 시 무료 자동 한국어 번역 및 실시간 지식베이스 추가 지원
"""

import json
import math
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import (
    FAISS_DB_DIR,
    PDF_FILES,
    build_vector_store,
    get_embeddings,
    load_and_enrich_pdfs,
    load_saved_api_key,
)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="실험·실습실 안전관리 AI 챗봇 (3모드 통합)",
    page_icon="🦺",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 로컬 Pre-Retrieval (Query Rewrite) 규칙 (0크레딧)
# ---------------------------------------------------------------------------
DOMAIN_REWRITE_RULES = [
    (
        ["ccc", "응급조치", "응급상황", "처리 과정", "처리과정", "사고 발생"],
        "실험·실습실 안전사고 발생 시 응급상황 처리 과정 4단계 및 응급조치 CCC(Check, Call, Care) 행동요령",
    ),
    (
        ["면장갑", "밀링", "띠톱", "선반", "절삭", "회전", "말림"],
        "선반·밀링·띠톱 등 회전체 기계 작업 시 면장갑 착용 금지 이유(회전말림점 위험) 및 금속 칩 제거 안전수칙",
    ),
    (
        ["황산", "산을 묽힐", "희석", "시약병", "누출", "산이나 알칼리"],
        "산(황산 등)을 묽히는 올바른 순서(물에 산을 천천히 투입) 및 황산액 시약병 파손·누출 시 20분 이상 세척 대처 방안",
    ),
    (
        ["일상점검", "정기점검", "특별안전점검", "정밀안전진단", "점검 주기", "점검주기", "실시주기"],
        "연구실안전법상 안전점검(일상점검 매일 1회, 정기점검 매년 1회, 특별안전점검) 및 정밀안전진단(2년마다 1회) 실시 주기",
    ),
    (
        ["화상", "데임", "물집", "뜨거운"],
        "실험·실습실 화상 발생 시 응급처치 요령(15~30분 찬물 냉각, 물집 보존) 및 화상 예방 안전수칙",
    ),
    (
        ["눈에", "세안기", "눈 세척", "눈세척", "칩이나 티"],
        "눈에 화학물질이나 칩·티가 들어갔을 때 눈 세척기(세안기) 15분 이상 세척 및 응급처치 요령",
    ),
    (
        ["고압멸균기", "오토클레이브", "autoclave", "가스크로마토", "gc", "무균실험대", "클린벤치"],
        "실험실 주요 기기(고압멸균기 Autoclave, 가스크로마토그래피 GC, 무균실험대 Clean bench) 위험요소 및 안전수칙",
    ),
    (
        ["소화기", "화재", "불이야", "119", "전기화재", "전기 화재", "감전"],
        "실험·실습실 화재·폭발 시 대피 및 소화기 사용요령과 감전·전기화재(C급 소화기, 물 분사 금지) 대처 방안",
    ),
    (
        ["가스", "아세틸렌", "수소", "산소", "용접", "도색", "색상"],
        "가스 실험·실습실 가연성·독성가스 안전관리, 가스 용기별 도색 색상 및 산소·아세틸렌 용접 안전수칙",
    ),
    (
        ["폐액", "폐기물", "폐시약", "혼합", "스티커", "80%"],
        "실험실 화학물질 폐기물·폐액 분류(산·중성·알칼리·할로겐) 및 안전관리(80% 초과 금지, 혼합 금지)",
    ),
]


def local_query_rewrite(query: str, history_messages: list[dict]) -> str:
    """API 크레딧 소모 없이 질문을 검색 최적화 문장으로 재작성합니다."""
    contextual = query
    if len(query) <= 16 and len(history_messages) >= 2:
        prev_user = next(
            (m["content"] for m in reversed(history_messages[:-1]) if m["role"] == "user"),
            "",
        )
        contextual = f"{prev_user} {query}".strip()

    q_lower = contextual.lower()
    for keywords, rewritten in DOMAIN_REWRITE_RULES:
        if any(k in q_lower for k in keywords):
            return f"{query} ({rewritten})"
    return f"{contextual} 실험·실습실 안전수칙 및 대처요령"


# ---------------------------------------------------------------------------
# 영어 텍스트 감지 및 무료 자동 번역
# ---------------------------------------------------------------------------
def is_mostly_english(text: str) -> bool:
    ascii_cnt = len(re.findall(r"[A-Za-z]", text))
    ko_cnt = len(re.findall(r"[가-힣]", text))
    return ascii_cnt > 25 and ascii_cnt > ko_cnt * 1.5


def translate_en_to_ko_free(text: str) -> str:
    """외부 LLM 크레딧을 쓰지 않고 영어 원문을 한국어로 자동 번역합니다."""
    try:
        trimmed = text[:1400]
        q = urllib.parse.quote(trimmed)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=ko&dt=t&q={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list) and isinstance(data[0], list):
                return "".join(part[0] or "" for part in data[0])
    except Exception:
        pass
    return text


# ---------------------------------------------------------------------------
# 기본 PDF 청크 로드 및 로컬 하이브리드 MMR 검색 (0크레딧 · 완전 오프라인)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_base_chunks() -> list[Document]:
    docs = load_and_enrich_pdfs()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(docs)


def tokenize_ko_en(text: str) -> list[str]:
    tokens = [t for t in re.split(r"[^\w가-힣]+", (text or "").lower()) if len(t) >= 2]
    expanded = list(tokens)
    for t in tokens:
        if len(t) >= 3:
            expanded.append(t[:2])
            expanded.append(t[:3])
    return expanded


def retrieve_local_mmr(
    chunks: list[Document],
    query: str,
    top_k: int = 3,
    search_type: str = "mmr",
) -> list[Document]:
    """임베딩 API 없이 100% 오프라인으로 가장 관련성 높은 청크를 검색합니다."""
    q_tokens = tokenize_ko_en(query)
    if not q_tokens:
        return chunks[:top_k]

    q_counts = Counter(q_tokens)
    scored: list[tuple[float, Document]] = []

    for doc in chunks:
        text_lower = doc.page_content.lower()
        score = 0.0
        for tok, q_freq in q_counts.items():
            if tok in text_lower:
                weight = 4.0 if len(tok) >= 3 else 1.5
                score += weight * (1.0 + math.log1p(q_freq))
        if "목 차" in doc.page_content or "CONTENTS" in doc.page_content:
            score *= 0.45
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    if search_type != "mmr":
        return [doc for _, doc in scored[:top_k]]

    candidates = scored[: max(15, top_k * 4)]
    selected: list[Document] = []
    while len(selected) < top_k and candidates:
        best_idx = 0
        best_mmr = -1e9
        for i, (base_score, cand_doc) in enumerate(candidates):
            penalty = 0.0
            c_file = cand_doc.metadata.get("file_name", "")
            c_page = cand_doc.metadata.get("page_number", 0)
            for sel_doc in selected:
                s_file = sel_doc.metadata.get("file_name", "")
                s_page = sel_doc.metadata.get("page_number", 0)
                if c_file == s_file and c_page == s_page:
                    penalty = max(penalty, 6.0)
                elif c_file == s_file and abs(c_page - s_page) <= 1:
                    penalty = max(penalty, 2.5)
            mmr_score = base_score - penalty
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i
        selected.append(candidates.pop(best_idx)[1])

    return selected


# ---------------------------------------------------------------------------
# 모드 1: ⚡ 오프라인 RAG 엔진 답변 합성기 (API 불필요 · 100% 오프라인)
# ---------------------------------------------------------------------------
def synthesize_offline_rag_answer(query: str, docs: list[Document]) -> str:
    """API 호출 없이 검색된 매뉴얼 청크에서 핵심 조항을 추출·구조화하여 답변을 생성합니다."""
    if not docs:
        return "질문하신 내용과 일치하는 안전수칙을 찾지 못했습니다. 다른 키워드(예: 화재, 화상, 밀링, 황산, 보호구, 안전점검)로 질문해 주세요."

    q_tokens = [t for t in re.split(r"[^\w가-힣]+", query.lower()) if len(t) >= 2]
    sections = [
        f"질문하신 **\"{query}\"**에 대해 실험·실습실 안전관리 매뉴얼에서 확인된 핵심 지침을 정리해 드립니다.\n"
    ]

    for idx, doc in enumerate(docs, 1):
        fname = doc.metadata.get("file_name", "안전매뉴얼")
        page = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
        raw_lines = [
            l.strip()
            for l in doc.page_content.splitlines()
            if len(l.strip()) > 2
            and not l.strip().startswith("[")
            and not re.match(r"^-?\s*\d+\s*-?$", l.strip())
            and "실습실(장) 안전관리 간편 매뉴얼" not in l
        ]
        if not raw_lines:
            continue

        heading = raw_lines[0].lstrip("❍▶⁍❖✚✓•-* ")
        if len(heading) > 55:
            heading = f"핵심 안전수칙 및 행동요령 ({fname} {page}쪽)"
        body_lines = raw_lines[1:] if len(raw_lines) > 1 else raw_lines

        bullets = []
        for line in body_lines[:10]:
            cleaned = line.lstrip("❍▶⁍❖✚✓•-* ").strip()
            if len(cleaned) < 3:
                continue
            for tok in q_tokens:
                if len(tok) >= 2 and tok in cleaned:
                    cleaned = cleaned.replace(tok, f"**{tok}**", 1)
                    break
            bullets.append(f"- {cleaned}")

        if bullets:
            sections.append(
                f"### {idx}. {heading}\n"
                + "\n".join(bullets)
                + f"\n> 📌 **출처:** [{fname} - {page}쪽]"
            )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# 모드 2: 🦙 무료 오픈소스 LLM 호출 (Ollama 로컬 실행 - 강의안 AdvancedRAG 반영)
# ---------------------------------------------------------------------------
def call_opensource_ollama_llm(
    ollama_url: str,
    ollama_model: str,
    temperature: float,
    context_text: str,
    history_text: str,
    question: str,
) -> str:
    """로컬 Ollama 서버(http://localhost:11434)의 무료 오픈소스 LLM을 호출합니다."""
    endpoint = f"{ollama_url.rstrip('/')}/api/chat"
    system_msg = (
        "너는 직업계고등학교 및 대학 실험·실습실 안전관리 전문 AI 도우미다.\n"
        "반드시 아래 [참고 문서]의 내용을 근거로 핵심 안전수칙과 절차를 한국어로 명확하게 항목별로 정리해 답하라.\n"
        "관련된 출처 문서명과 페이지 번호도 함께 밝혀라.\n\n"
        f"[참고 문서]\n{context_text}"
    )
    user_msg = f"[이전 대화]\n{history_text}\n\n[질문]\n{question}"
    payload = json.dumps(
        {
            "model": ollama_model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("message", {}).get("content", "").strip()


# ---------------------------------------------------------------------------
# 모드 3: ☁️ 클라우드 API 호출 (초경량 모델 + 동일 질문 캐싱)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def generate_cached_llm_answer(
    api_key: str,
    primary_model: str,
    temperature: float,
    context_text: str,
    history_text: str,
    question: str,
) -> tuple[str, str]:
    """초경량 Gemini 모델로 답변을 생성하고 동일 질문을 캐싱하여 크레딧을 절약합니다."""
    candidate_models = [primary_model]
    for fallback_m in [
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]:
        if fallback_m not in candidate_models:
            candidate_models.append(fallback_m)

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "너는 직업계고등학교 및 대학 실험·실습실 안전관리 전문 AI 도우미다.\n"
                "반드시 아래 [참고 문서]의 내용만 근거로 핵심 안전수칙과 절차를 간결하고 명확하게 항목별로 정리해 답하라.\n"
                "영어 원문과 한국어 번역이 함께 있는 경우 자연스러운 한국어로 설명하고, "
                "관련된 출처 문서명과 페이지 번호(예: [직업계고등학교 실습실 안전관리 간편 매뉴얼 9쪽])를 함께 밝혀라.\n\n"
                "[참고 문서]\n{context}",
            ),
            (
                "human",
                "[이전 대화]\n{chat_history}\n\n[질문]\n{question}",
            ),
        ]
    )

    last_err = None
    for m_name in candidate_models:
        try:
            llm = ChatGoogleGenerativeAI(
                model=m_name,
                temperature=temperature,
                max_output_tokens=700,
                google_api_key=api_key,
            )
            chain = rag_prompt | llm | StrOutputParser()
            ans = chain.invoke(
                {
                    "context": context_text,
                    "chat_history": history_text,
                    "question": question,
                }
            )
            return ans, m_name
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"API 호출 실패: {last_err}")


def format_chat_history(messages: list[dict], max_turns: int = 2) -> str:
    recent = messages[-(max_turns * 2) :]
    if not recent:
        return "없음"
    lines = []
    for m in recent:
        role_kr = "사용자" if m["role"] == "user" else "AI"
        lines.append(f"{role_kr}: {m['content'][:200]}")
    return "\n".join(lines)


def format_docs_compact(docs: list[Document], max_chars_per_doc: int = 650) -> str:
    formatted = []
    for i, doc in enumerate(docs, 1):
        fname = doc.metadata.get("file_name", Path(doc.metadata.get("source", "")).name)
        page = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
        snippet = doc.page_content.strip()[:max_chars_per_doc]
        formatted.append(f"[문서 {i} | 출처: {fname} ({page}쪽)]\n{snippet}")
    return "\n\n---\n\n".join(formatted)


# ---------------------------------------------------------------------------
# 사이드바: 3가지 엔진 모드 선택 + RAG 파라미터 + 영어 PDF 번역 업로드
# ---------------------------------------------------------------------------
if "extra_chunks" not in st.session_state:
    st.session_state.extra_chunks = []
if "uploaded_pdf_names" not in st.session_state:
    st.session_state.uploaded_pdf_names = []

with st.sidebar:
    st.header("🎛️ 챗봇 엔진 모드 선택")

    engine_mode = st.radio(
        "사용할 챗봇 버전을 선택하세요",
        options=["offline_rag", "opensource_llm", "cloud_api"],
        index=0,
        format_func=lambda x: {
            "offline_rag": "⚡ 1. 오프라인 RAG 엔진 (API 불필요 · 0원)",
            "opensource_llm": "🦙 2. 무료 오픈소스 LLM (Ollama · 0원)",
            "cloud_api": "☁️ 3. API 사용 버전 (Gemini 초경량)",
        }[x],
    )

    api_key_input = ""
    model_name = "gemini-2.5-flash-lite"
    ollama_model = "qwen2.5:3b"
    ollama_url = "http://localhost:11434"
    temperature = 0.1

    if engine_mode == "offline_rag":
        st.success("✅ **오프라인 RAG 모드 활성**: API 키나 외부 서버 없이 내장된 안전매뉴얼 RAG 엔진으로 즉시 답변합니다.")

    elif engine_mode == "opensource_llm":
        st.info(
            "🦙 **무료 오픈소스 LLM 모드 (강의안 AdvancedRAG)**: 내 PC에 설치된 Ollama 오픈소스 모델을 호출합니다. (미실행 시 오프라인 RAG 엔진으로 자동 전환)"
        )
        ollama_model = st.selectbox(
            "오픈소스 LLM 모델 선택",
            options=[
                "qwen2.5:3b",
                "exaone3.5:2.4b",
                "gemma2:2b",
                "llama3.2:3b",
            ],
            index=0,
            help="터미널에서 `ollama run qwen2.5:3b` 명령으로 무료 실행 가능한 한국어 지원 오픈소스 모델입니다.",
        )
        ollama_url = st.text_input("Ollama 로컬 주소", value="http://localhost:11434")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)

    elif engine_mode == "cloud_api":
        env_key = load_saved_api_key()
        if env_key:
            st.success("🔒 서버에 Gemini API 키가 안전하게 연결됨")
            custom_key = st.text_input(
                "🔑 다른 API 키 직접 사용 (선택)",
                value="",
                type="password",
                placeholder="비워두면 서버 설정 키 사용",
            )
            api_key_input = custom_key.strip() or env_key
        else:
            api_key_input = st.text_input(
                "🔑 Gemini API Key 입력",
                value="",
                type="password",
                placeholder="AIzaSy...",
            ).strip()

        model_name = st.selectbox(
            "Gemini 모델 (위에서부터 크레딧 소모 최소)",
            options=[
                "gemini-2.5-flash-lite",
                "gemini-2.0-flash-lite",
                "gemini-2.5-flash",
                "gemini-2.0-flash",
            ],
            index=0,
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)

    st.divider()
    st.subheader("🔍 리트리버(검색) 설정")
    search_type = st.radio(
        "검색 방식 (Retriever)",
        options=["mmr", "similarity"],
        index=0,
        format_func=lambda x: "MMR (다양성 + 관련성 균형)" if x == "mmr" else "Similarity (단순 유사도)",
    )
    top_k = st.slider("검색할 문서 조각 수 (top-k)", 2, 6, 3)
    use_rewrite = st.checkbox("Pre-Retrieval: 질문 재작성 (0크레딧 로컬)", value=True)

    st.divider()
    st.subheader("🌐 영어/추가 PDF 업로드 (자동 한글 번역)")
    uploaded_files = st.file_uploader(
        "영어 원문 PDF를 올리면 자동으로 한국어로 번역해 지식베이스에 추가합니다.",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
        for uf in uploaded_files:
            if uf.name not in st.session_state.uploaded_pdf_names:
                with st.spinner(f"[{uf.name}] 추출 및 한국어 자동 번역 중..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uf.getbuffer())
                        tmp_path = tmp.name
                    try:
                        loader = PyPDFLoader(tmp_path)
                        raw_docs = loader.load()
                        new_docs = []
                        for d in raw_docs:
                            pno = d.metadata.get("page", 0) + 1
                            txt = (d.page_content or "").strip()
                            if len(txt) < 10:
                                continue
                            if is_mostly_english(txt):
                                ko_txt = translate_en_to_ko_free(txt)
                                d.page_content = (
                                    f"[자동 한국어 번역 ({uf.name} - {pno}쪽)]\n{ko_txt}\n\n"
                                    f"[영어 원문]\n{txt}"
                                )
                            else:
                                d.page_content = f"[{uf.name} - {pno}쪽]\n{txt}"
                            d.metadata["file_name"] = uf.name
                            d.metadata["page_number"] = pno
                            new_docs.append(d)
                        split_new = splitter.split_documents(new_docs)
                        st.session_state.extra_chunks.extend(split_new)
                        st.session_state.uploaded_pdf_names.append(uf.name)
                        st.success(f"✅ {uf.name}: {len(split_new)}개 청크 자동 번역 & 추가 완료!")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

    st.divider()
    st.subheader("📚 학습된 문서 목록")
    for pdf in PDF_FILES:
        st.caption(f"✅ `{pdf.name}`")
    for up_name in st.session_state.uploaded_pdf_names:
        st.caption(f"🌐 `{up_name}` (번역 완료)")

    if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# 메인 화면
# ---------------------------------------------------------------------------
mode_labels = {
    "offline_rag": "⚡ 오프라인 RAG 엔진 버전 (API 불필요 · 0원)",
    "opensource_llm": f"🦙 무료 오픈소스 LLM 버전 (Ollama: {ollama_model})",
    "cloud_api": f"☁️ 클라우드 API 버전 ({model_name} 최소 크레딧)",
}

st.title("🦺 실험·실습실 안전관리 AI 챗봇")
st.caption(f"현재 선택된 모드: **{mode_labels[engine_mode]}** | 왼쪽 사이드바에서 언제든 버전을 전환할 수 있습니다.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "안녕하세요! **직업계고등학교 실습실 안전관리 간편 매뉴얼**과 "
                "**[한국교육시설안전원] 연구활동종사자 실험실습실 안전교육** 문서를 학습한 AI 안전도우미입니다.\n\n"
                "왼쪽 사이드바에서 **① ⚡ 오프라인 RAG 엔진 버전**, **② 🦙 무료 오픈소스 LLM 버전(Ollama)**, **③ ☁️ API 사용 버전(Gemini Flash-Lite)** 중 원하는 모드를 자유롭게 선택해 질문해 보세요!"
            ),
        }
    ]

sample_questions = [
    "실습실에서 안전사고 발생 시 응급조치 CCC가 무엇인가요?",
    "선반·밀링이나 띠톱 작업할 때 면장갑을 끼면 안 되는 이유는?",
    "산(황산 등)을 묽힐 때나 황산액이 누출되었을 때 어떻게 대처해야 해?",
    "연구실 일상점검, 정기점검, 정밀안전진단의 실시 주기는 어떻게 되나요?",
]

cols = st.columns(len(sample_questions))
selected_sample = None
for col, sq in zip(cols, sample_questions):
    if col.button(f"💡 {sq}", use_container_width=True):
        selected_sample = sq

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("engine_badge"):
            st.caption(f"🎛️ **응답 모드:** `{msg['engine_badge']}`")
        if msg.get("rewritten_query"):
            st.caption(f"🔍 **Query Rewrite 검색어:** `{msg['rewritten_query']}`")
        if msg.get("sources"):
            with st.expander("📖 참고한 매뉴얼 출처 및 원문 보기"):
                for s in msg["sources"]:
                    st.markdown(f"**• {s['title']}**")
                    st.caption(s["preview"])

user_input = st.chat_input("실험·실습실 안전 수칙이나 업로드한 영어 PDF 내용을 질문하세요...")
query = selected_sample or user_input

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        all_chunks = get_base_chunks() + st.session_state.extra_chunks
        history_text = format_chat_history(st.session_state.messages[:-1])

        # 1. Pre-Retrieval (Query Rewrite)
        search_query = local_query_rewrite(query, st.session_state.messages) if use_rewrite else query

        # 2. Retrieval (로컬 하이브리드 MMR)
        retrieved_docs = retrieve_local_mmr(all_chunks, search_query, top_k, search_type)
        context_text = format_docs_compact(retrieved_docs)

        if use_rewrite and search_query != query:
            st.caption(f"🔍 **Query Rewrite 검색어:** `{search_query}`")

        # 3. 선택된 모드별 답변 생성
        used_badge = ""
        if engine_mode == "offline_rag":
            answer = synthesize_offline_rag_answer(query, retrieved_docs)
            used_badge = "⚡ 오프라인 RAG 엔진 (0크레딧)"

        elif engine_mode == "opensource_llm":
            with st.spinner(f"🦙 무료 오픈소스 LLM ({ollama_model}) 호출 중..."):
                try:
                    answer = call_opensource_ollama_llm(
                        ollama_url=ollama_url,
                        ollama_model=ollama_model,
                        temperature=temperature,
                        context_text=context_text,
                        history_text=history_text,
                        question=query,
                    )
                    used_badge = f"🦙 무료 오픈소스 LLM (Ollama: {ollama_model})"
                except Exception:
                    st.info(
                        f"💡 로컬 Ollama 서버(`{ollama_url}`)가 실행 중이지 않아 **⚡ 오프라인 RAG 엔진**으로 즉시 답변해 드립니다. "
                        f"(Ollama 사용 시 터미널에서 `ollama run {ollama_model}` 실행)"
                    )
                    answer = synthesize_offline_rag_answer(query, retrieved_docs)
                    used_badge = "⚡ 오프라인 RAG 엔진 (Ollama 미실행 자동 전환)"

        else:  # cloud_api
            if not api_key_input:
                st.info("💡 API 키가 입력되지 않아 **⚡ 오프라인 RAG 엔진**으로 즉시 답변해 드립니다.")
                answer = synthesize_offline_rag_answer(query, retrieved_docs)
                used_badge = "⚡ 오프라인 RAG 엔진 (API 키 미입력 자동 전환)"
            else:
                with st.spinner(f"☁️ 초경량 API 모델({model_name})로 답변 생성 중..."):
                    try:
                        answer, used_m = generate_cached_llm_answer(
                            api_key=api_key_input,
                            primary_model=model_name,
                            temperature=temperature,
                            context_text=context_text,
                            history_text=history_text,
                            question=query,
                        )
                        used_badge = f"☁️ 클라우드 API ({used_m} · 최소 크레딧)"
                    except Exception as e:
                        st.warning(f"API 호출 한도/오류({e})가 발생하여 **⚡ 오프라인 RAG 엔진**으로 자동 전환합니다.")
                        answer = synthesize_offline_rag_answer(query, retrieved_docs)
                        used_badge = "⚡ 오프라인 RAG 엔진 (API 폴백)"

        st.markdown(answer)
        st.caption(f"🎛️ **응답 모드:** `{used_badge}` | **검색 청크:** `{len(retrieved_docs)}개`")

        sources_info = []
        for doc in retrieved_docs:
            fname = doc.metadata.get("file_name", Path(doc.metadata.get("source", "")).name)
            page = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
            preview = doc.page_content.strip().replace("\n", " ")[:240] + "..."
            sources_info.append({"title": f"{fname} ({page}쪽)", "preview": preview})

        with st.expander("📖 참고한 매뉴얼 출처 및 원문 보기"):
            for s in sources_info:
                st.markdown(f"**• {s['title']}**")
                st.caption(s["preview"])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "engine_badge": used_badge,
                "rewritten_query": search_query if (use_rewrite and search_query != query) else None,
                "sources": sources_info,
            }
        )
