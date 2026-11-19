"""
[Step 2] 실험·실습실 안전관리 RAG 챗봇 (Streamlit UI - 초저비용/최소 크레딧 최적화 버전)
전주대학교 AI융합프로젝트 (김영수 교수님) 강의안 전 과정 통합 구현:
- 크레딧 최적화 1: 가장 저렴한 초경량 모델(gemini-2.5-flash-lite / gemini-2.0-flash-lite) 기본 적용
- 크레딧 최적화 2: Pre-Retrieval(Query Rewrite)을 로컬 안전 도메인 규칙으로 처리하여 질문당 LLM 호출을 2회 → 1회로 50% 절감
- 크레딧 최적화 3: 로컬 하이브리드(BM25 + MMR) 리트리버를 기본 사용하여 임베딩 API 크레딧 소모 0원 (FAISS 임베딩도 선택 가능)
- 크레딧 최적화 4: top_k=3, 청크 길이 압축, 최근 대화 2턴 제한, max_output_tokens 제한 및 동일 질문 캐싱(@st.cache_data)
- 영어 원문 PDF 업로드 시 무료 자동 한국어 번역 및 실시간 지식베이스 추가 지원
"""

import json
import math
import os
import re
import tempfile
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
    page_title="실험·실습실 안전관리 AI 챗봇 (크레딧 최적화)",
    page_icon="🦺",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 로컬 Pre-Retrieval (Query Rewrite) 규칙 (LLM 추가 호출 없이 0크레딧으로 검색어 최적화)
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
    """API 크레딧을 전혀 쓰지 않고 질문을 검색 최적화 문장으로 재작성합니다."""
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
# 영어 텍스트 감지 및 무료 자동 번역 (API 크레딧 0원 소모)
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
# 기본 PDF 청크 로드 (메모리 캐싱 - 임베딩 API 호출 없이 즉시 로드)
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
    """임베딩 API 크레딧 소모 없이 로컬 TF-IDF/BM25 + MMR 방식으로 가장 관련도 높은 청크를 검색합니다."""
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


def synthesize_fallback_answer(query: str, docs: list[Document]) -> str:
    """API 키가 없거나 한도 초과일 때도 매뉴얼 원문을 바탕으로 깔끔한 구조화 답변을 생성합니다."""
    if not docs:
        return "질문하신 내용과 일치하는 안전수칙을 찾지 못했습니다."
    lines = [f"질문하신 **'{query}'**에 대한 실험·실습실 안전관리 매뉴얼 핵심 지침입니다.\n"]
    for idx, doc in enumerate(docs, 1):
        fname = doc.metadata.get("file_name", "안전매뉴얼")
        page = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
        body = [
            l.strip().lstrip("❍▶⁍❖✚✓•-* ")
            for l in doc.page_content.splitlines()
            if len(l.strip()) > 2 and not l.strip().startswith("[")
        ]
        bullets = "\n".join(f"- {b}" for b in body[:8])
        lines.append(f"### {idx}. {fname} ({page}쪽)\n{bullets}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# 사이드바: 크레딧 절약 설정 & 영어 PDF 자동 번역 업로드
# ---------------------------------------------------------------------------
if "extra_chunks" not in st.session_state:
    st.session_state.extra_chunks = []
if "uploaded_pdf_names" not in st.session_state:
    st.session_state.uploaded_pdf_names = []

with st.sidebar:
    st.header("⚙️ RAG 챗봇 설정 (초저비용 최적화)")

    env_key = load_saved_api_key()
    if env_key:
        st.success("🔒 Gemini API 키가 안전하게 연결됨")
        custom_key = st.text_input(
            "🔑 다른 API 키 직접 사용 (선택)",
            value="",
            type="password",
            placeholder="비워두면 서버 설정 키 사용",
        )
        api_key_input = custom_key.strip() or env_key
    else:
        api_key_input = st.text_input(
            "🔑 Gemini API Key (선택)",
            value="",
            type="password",
            placeholder="비워두어도 내장 RAG 요약이 즉시 동작합니다",
        ).strip()

    st.divider()
    st.subheader("💰 최소 크레딧 모델 & 절약 옵션")
    model_name = st.selectbox(
        "Gemini 모델 (위에서부터 크레딧 소모 최소)",
        options=[
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
        ],
        index=0,
        help="flash-lite 모델은 일반 모델 대비 크레딧 소모가 가장 적고 무료 호출 한도가 가장 넉넉합니다.",
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)

    retriever_engine = st.radio(
        "임베딩/검색 엔진",
        options=["local_zero_credit", "faiss_gemini"],
        index=0,
        format_func=lambda x: (
            "⚡ 로컬 하이브리드 MMR (임베딩 크레딧 0원 · 권장)"
            if x == "local_zero_credit"
            else "🧠 FAISS + Gemini 임베딩 API 사용"
        ),
    )

    search_type = st.radio(
        "검색 방식 (Retriever)",
        options=["mmr", "similarity"],
        index=0,
        format_func=lambda x: "MMR (다양성 + 관련성 균형)" if x == "mmr" else "Similarity (단순 유사도)",
    )
    top_k = st.slider("검색할 문서 조각 수 (k - 적을수록 토큰 절약)", 2, 6, 3)

    rewrite_mode = st.selectbox(
        "Pre-Retrieval (Query Rewrite 방식)",
        options=["local_zero_credit", "llm_rewrite", "off"],
        index=0,
        format_func=lambda x: {
            "local_zero_credit": "⚡ 로컬 안전규칙 재작성 (크레딧 0원 · 권장)",
            "llm_rewrite": "🤖 LLM 질문 재작성 (질문당 API 1회 추가 소모)",
            "off": "사용 안 함",
        }[x],
    )

    st.divider()
    st.subheader("🌐 영어/추가 PDF 업로드 (자동 한글 번역)")
    uploaded_files = st.file_uploader(
        "영어 원문 PDF를 올리면 자동으로 한국어로 번역해 챗봇에 추가합니다 (LLM 크레딧 0원)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
        for uf in uploaded_files:
            if uf.name not in st.session_state.uploaded_pdf_names:
                with st.spinner(f"[{uf.name}] 텍스트 추출 및 한국어 자동 번역 중..."):
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
    st.subheader("📚 현재 학습된 문서")
    for pdf in PDF_FILES:
        st.caption(f"✅ `{pdf.name}`")
    for up_name in st.session_state.uploaded_pdf_names:
        st.caption(f"🌐 `{up_name}` (자동 번역 완료)")

    if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# 동일 질문 응답 캐싱 (같은 질문 반복 시 API 크레딧 0원 소모)
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
    """
    가장 크레딧 소모가 적은 모델로 답변을 생성하고 캐싱합니다.
    선택한 모델이 일시적 한도 초과나 미지원일 경우 자동으로 차순위 경량 모델로 폴백합니다.
    """
    candidate_models = [primary_model]
    for fallback_m in ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]:
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
    raise RuntimeError(f"LLM 호출 실패: {last_err}")


def format_chat_history(messages: list[dict], max_turns: int = 2) -> str:
    """토큰 절약을 위해 최근 2턴의 대화만 압축하여 전달합니다."""
    recent = messages[-(max_turns * 2) :]
    if not recent:
        return "없음"
    lines = []
    for m in recent:
        role_kr = "사용자" if m["role"] == "user" else "AI"
        # 이전 답변은 200자까지만 요약해 입력 토큰 절약
        short_content = m["content"][:200]
        lines.append(f"{role_kr}: {short_content}")
    return "\n".join(lines)


def format_docs_compact(docs: list[Document], max_chars_per_doc: int = 650) -> str:
    """입력 토큰(크레딧) 소모를 최소화하기 위해 검색된 청크 핵심 내용을 압축 포맷팅합니다."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        fname = doc.metadata.get("file_name", Path(doc.metadata.get("source", "")).name)
        page = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
        snippet = doc.page_content.strip()[:max_chars_per_doc]
        formatted.append(f"[문서 {i} | 출처: {fname} ({page}쪽)]\n{snippet}")
    return "\n\n---\n\n".join(formatted)


# ---------------------------------------------------------------------------
# 메인 화면
# ---------------------------------------------------------------------------
st.title("🦺 실험·실습실 안전관리 AI 챗봇")
st.caption(
    "전주대학교 AI융합프로젝트 · 초경량 모델(Flash-Lite) + 0크레딧 로컬 하이브리드 검색 + 영어 PDF 자동 번역 지원"
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "안녕하세요! **직업계고등학교 실습실 안전관리 간편 매뉴얼**과 "
                "**[한국교육시설안전원] 연구활동종사자 실험실습실 안전교육** 문서를 학습한 AI 안전도우미입니다.\n\n"
                "- 💰 **크레딧 최소 소모 최적화**: 초경량 `flash-lite` 모델 + 0크레딧 로컬 MMR 검색 + 동일 질문 캐싱이 적용되어 있습니다.\n"
                "- 🌐 **영어 PDF 자동 번역**: 왼쪽 사이드바에 영어 원문 PDF를 올리면 자동으로 한국어로 번역해 답변합니다!"
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
        search_query = query
        if rewrite_mode == "local_zero_credit":
            search_query = local_query_rewrite(query, st.session_state.messages)
        elif rewrite_mode == "llm_rewrite" and api_key_input:
            try:
                llm_rw = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.0,
                    max_output_tokens=80,
                    google_api_key=api_key_input,
                )
                rw_prompt = ChatPromptTemplate.from_template(
                    "다음 질문을 실험실 안전매뉴얼 검색용 한국어 한 문장으로만 재작성하라: {q}"
                )
                search_query = (rw_prompt | llm_rw | StrOutputParser()).invoke({"q": query}).strip()
            except Exception:
                search_query = local_query_rewrite(query, st.session_state.messages)

        # 2. Retrieval (0크레딧 로컬 하이브리드 MMR 기본, 선택 시 FAISS 사용)
        if retriever_engine == "faiss_gemini" and api_key_input:
            try:
                emb = get_embeddings(api_key=api_key_input)
                if not (FAISS_DB_DIR / "index.faiss").exists():
                    vs = build_vector_store(api_key=api_key_input)
                else:
                    vs = FAISS.load_local(
                        str(FAISS_DB_DIR), emb, allow_dangerous_deserialization=True
                    )
                retriever = vs.as_retriever(
                    search_type=search_type,
                    search_kwargs={"k": top_k},
                )
                retrieved_docs = retriever.invoke(search_query)
            except Exception:
                retrieved_docs = retrieve_local_mmr(all_chunks, search_query, top_k, search_type)
        else:
            retrieved_docs = retrieve_local_mmr(all_chunks, search_query, top_k, search_type)

        context_text = format_docs_compact(retrieved_docs)

        if rewrite_mode != "off" and search_query != query:
            st.caption(f"🔍 **Query Rewrite 검색어 (0크레딧 로컬 최적화):** `{search_query}`")

        # 3. Generation (캐싱된 초경량 LLM 호출 또는 내장 요약 폴백)
        used_model_badge = "내장 RAG 요약 엔진 (0크레딧)"
        if api_key_input:
            with st.spinner(f"초경량 모델({model_name})로 답변 생성 중..."):
                try:
                    answer, used_m = generate_cached_llm_answer(
                        api_key=api_key_input,
                        primary_model=model_name,
                        temperature=temperature,
                        context_text=context_text,
                        history_text=history_text,
                        question=query,
                    )
                    used_model_badge = f"{used_m} (최소 크레딧 모드)"
                except Exception:
                    answer = synthesize_fallback_answer(query, retrieved_docs)
        else:
            answer = synthesize_fallback_answer(query, retrieved_docs)

        st.markdown(answer)
        st.caption(f"⚡ **응답 엔진:** `{used_model_badge}` | **검색 청크:** `{len(retrieved_docs)}개`")

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
                "rewritten_query": search_query if (rewrite_mode != "off" and search_query != query) else None,
                "sources": sources_info,
            }
        )
