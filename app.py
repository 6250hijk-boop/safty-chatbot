"""
[Step 2] 실험·실습실 안전관리 RAG 챗봇 (Streamlit UI)
전주대학교 AI융합프로젝트 (김영수 교수님) 강의안 전 과정 통합 구현:
- 01-03: RAG 파이프라인 & 환경변수(.env) 로드
- 04-07: ChatPromptTemplate · ChatGoogleGenerativeAI · StrOutputParser · LCEL 체인 & stream()
- 09-11: PyPDFLoader · RecursiveCharacterTextSplitter · GoogleGenerativeAIEmbeddings
- 12-13-08: FAISS 벡터 스토어 · MMR Retriever · Streamlit 챗봇 UI (session_state, chat_message, write_stream)
- Advanced RAG: 대화 메모리(Multi-turn History) · Pre-Retrieval(Query Rewrite 질문 재작성) · 출처 페이지 표기
"""

import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ingest import FAISS_DB_DIR, PDF_FILES, build_vector_store, get_embeddings, load_saved_api_key

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="실험·실습실 안전관리 AI 챗봇",
    page_icon="🦺",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 사이드바: API 키 설정, 모델/리트리버 옵션, 벡터 스토어 관리
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ RAG 챗봇 설정")

    env_key = load_saved_api_key()
    api_key_input = st.text_input(
        "🔑 Gemini API Key",
        value=env_key,
        type="password",
        placeholder="AIzaSy...",
        help="api_key.txt(.env) 파일에 GEMINI_API_KEY를 설정하거나 여기에 직접 입력하세요.",
    )

    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        os.environ["GOOGLE_API_KEY"] = api_key_input

    st.divider()
    st.subheader("🤖 모델 및 RAG 설정")
    model_name = st.selectbox(
        "Gemini 모델 선택",
        options=[
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-pro",
        ],
        index=0,
    )
    temperature = st.slider("Temperature (창의성 조절)", 0.0, 1.0, 0.2, 0.1)

    search_type = st.radio(
        "검색 방식 (Retriever)",
        options=["mmr", "similarity"],
        index=0,
        format_func=lambda x: "MMR (다양성 + 관련성 균형)" if x == "mmr" else "Similarity (단순 유사도 상위)",
    )
    top_k = st.slider("검색할 문서 조각 수 (k)", 2, 8, 5)
    use_query_rewrite = st.checkbox(
        "Pre-Retrieval: 질문 재작성 (Query Rewrite)",
        value=True,
        help="Advanced RAG 기법: 사용자의 일상 질문을 검색에 최적화된 명확한 질문으로 먼저 다듬은 뒤 벡터 검색을 수행합니다.",
    )

    st.divider()
    st.subheader("📚 지식베이스 문서 (PDF)")
    for pdf in PDF_FILES:
        exists_badge = "✅" if pdf.exists() else "❌"
        st.caption(f"{exists_badge} `{pdf.name}`")

    index_exists = (FAISS_DB_DIR / "index.faiss").exists()
    if index_exists:
        st.success("FAISS 벡터 DB 준비 완료 (`./db/faiss`)")
    else:
        st.warning("벡터 DB가 아직 생성되지 않았습니다.")

    if st.button("🔄 벡터 스토어 생성 / 재구축", use_container_width=True):
        if not api_key_input:
            st.error("먼저 Gemini API Key를 입력해주세요!")
        else:
            with st.spinner("PDF 로드 → 청킹 → Gemini 임베딩 → FAISS 저장 중..."):
                try:
                    build_vector_store(api_key=api_key_input)
                    st.cache_resource.clear()
                    st.success("FAISS 벡터 스토어 구축 완료!")
                    st.rerun()
                except Exception as e:
                    st.error(f"벡터 스토어 생성 실패: {e}")

    if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# FAISS 벡터 스토어 로드 (캐싱)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_faiss_db(api_key: str):
    emb = get_embeddings(api_key=api_key)
    if not (FAISS_DB_DIR / "index.faiss").exists():
        return build_vector_store(api_key=api_key)
    return FAISS.load_local(
        str(FAISS_DB_DIR),
        emb,
        allow_dangerous_deserialization=True,
    )


def format_chat_history(messages: list[dict], max_turns: int = 4) -> str:
    """최근 대화 기록을 문자열로 포맷팅합니다 (Advanced RAG 멀티턴 메모리)."""
    recent = messages[-(max_turns * 2) :]
    if not recent:
        return "이전 대화 없음"
    lines = []
    for m in recent:
        role_kr = "사용자" if m["role"] == "user" else "AI"
        lines.append(f"{role_kr}: {m['content']}")
    return "\n".join(lines)


def format_docs(docs: list) -> str:
    """검색된 Document 리스트를 프롬프트 컨텍스트 문자열로 결합합니다."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        fname = doc.metadata.get("file_name", Path(doc.metadata.get("source", "")).name)
        page = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
        formatted.append(f"[문서 {i} | 출처: {fname} ({page}쪽)]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(formatted)


# ---------------------------------------------------------------------------
# 메인 화면 헤더 및 추천 질문 버튼
# ---------------------------------------------------------------------------
st.title("🦺 실험·실습실 안전관리 AI 챗봇")
st.caption(
    "전주대학교 AI융합프로젝트 · LangChain + Gemini + FAISS + Advanced RAG 기반 실습실 안전 매뉴얼 질의응답 시스템"
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "안녕하세요! **직업계고등학교 실습실 안전관리 간편 매뉴얼**과 "
                "**[한국교육시설안전원] 연구활동종사자 실험실습실 안전교육** 문서를 학습한 AI 안전도우미입니다.\n\n"
                "궁금한 안전수칙, 보호구 착용 기준, 기기 조작 주의사항, 응급처치 요령 등을 질문해 주세요!"
            ),
        }
    ]

# 추천 질문 빠른 실행
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

# 기존 대화 출력
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

# ---------------------------------------------------------------------------
# 사용자 질문 입력 및 RAG 체인 실행
# ---------------------------------------------------------------------------
user_input = st.chat_input("실험·실습실 안전 수칙이나 응급처치 방법을 질문하세요...")
query = selected_sample or user_input

if query:
    if not api_key_input:
        st.warning("왼쪽 사이드바에 **Gemini API Key**를 입력하거나 `.env` 파일에 설정해 주세요.")
        st.stop()

    # 1. 사용자 메시지 저장 및 화면 표시
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. AI 답변 생성 (Pre-Retrieval -> Retrieval -> LCEL Generation Stream)
    with st.chat_message("assistant"):
        try:
            with st.spinner("안전 매뉴얼 검색 및 답변 준비 중..."):
                vs = load_faiss_db(api_key_input)

                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    google_api_key=api_key_input,
                )

                history_text = format_chat_history(st.session_state.messages[:-1])

                # [Advanced RAG - Pre-Retrieval] Query Rewrite 체인 (강의안 AdvancedRAG p.21 & 04-07 LCEL)
                search_query = query
                if use_query_rewrite:
                    rewrite_prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "너는 검색 최적화 전문가다. 이전 대화 맥락을 참고하여 사용자의 질문을 "
                                "실험·실습실 안전관리 매뉴얼 벡터 검색에 가장 적합한 명확한 한국어 검색 질문 한 문장으로만 재작성하라. "
                                "부연 설명 없이 재작성된 질문만 출력하라.",
                            ),
                            (
                                "human",
                                "[이전 대화]\n{chat_history}\n\n[사용자 질문]\n{question}",
                            ),
                        ]
                    )
                    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
                    rewritten = rewrite_chain.invoke(
                        {"chat_history": history_text, "question": query}
                    ).strip()
                    if rewritten:
                        search_query = rewritten

                # [Retrieval] MMR 또는 Similarity 리트리버 검색 (강의안 12-13-08 p.18~19)
                if search_type == "mmr":
                    retriever = vs.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": top_k, "fetch_k": max(20, top_k * 5)},
                    )
                else:
                    retriever = vs.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": top_k},
                    )

                retrieved_docs = retriever.invoke(search_query)
                context_text = format_docs(retrieved_docs)

                # [Generation] LCEL 프롬프트 + 모델 + 출력파서 체인 (강의안 04-07 & 12-13-08)
                rag_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "너는 직업계고등학교 및 대학 실험·실습실 안전관리 전문 AI 도우미다.\n"
                            "반드시 아래 제공된 [참고 문서]의 내용을 근거로 정확하고 친절하게 답변하라.\n"
                            "답변할 때는 핵심 안전수칙이나 절차를 읽기 쉽게 항목별로 정리하고, "
                            "관련된 출처 문서명과 페이지 번호(예: [직업계고등학교 실습실 안전관리 간편 매뉴얼 9쪽])를 함께 밝혀라.\n"
                            "만약 [참고 문서]에 없는 내용이라면 추측해서 지어내지 말고, 매뉴얼에서 확인되지 않는 내용이라고 솔직히 안내하라.\n\n"
                            "[참고 문서]\n{context}",
                        ),
                        (
                            "human",
                            "[이전 대화 기록]\n{chat_history}\n\n[질문]\n{question}",
                        ),
                    ]
                )

                rag_chain = rag_prompt | llm | StrOutputParser()

            if use_query_rewrite and search_query != query:
                st.caption(f"🔍 **Query Rewrite 검색어:** `{search_query}`")

            # [Streaming 출력] 강의안 12-13-08 p.28 st.write_stream 활용
            answer = st.write_stream(
                rag_chain.stream(
                    {
                        "context": context_text,
                        "chat_history": history_text,
                        "question": query,
                    }
                )
            )

            # 출처 정보 정리 및 표시
            sources_info = []
            for doc in retrieved_docs:
                fname = doc.metadata.get("file_name", Path(doc.metadata.get("source", "")).name)
                page = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
                preview = doc.page_content.strip().replace("\n", " ")[:220] + "..."
                sources_info.append(
                    {
                        "title": f"{fname} ({page}쪽)",
                        "preview": preview,
                    }
                )

            with st.expander("📖 참고한 매뉴얼 출처 및 원문 보기"):
                for s in sources_info:
                    st.markdown(f"**• {s['title']}**")
                    st.caption(s["preview"])

            # 세션 상태에 AI 응답 저장
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "rewritten_query": search_query if (use_query_rewrite and search_query != query) else None,
                    "sources": sources_info,
                }
            )

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")

