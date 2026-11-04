"""
[Step 1] 문서 수집 · 청킹 · 임베딩 · FAISS 벡터 스토어 저장 (Indexing 파이프라인)
전주대학교 AI융합프로젝트 (김영수 교수님) 강의안 09-11, 12-13-08 기반 구현
"""

import logging
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from pdf_ocr_fallback import HIGHSCHOOL_PDF_FALLBACK, KOIES_PDF_FALLBACK

# pypdf 내부 PDF 구조 경고 로그 숨김
logging.getLogger("pypdf").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent
FAISS_DB_DIR = BASE_DIR / "db" / "faiss"

PDF_FILES = [
    BASE_DIR / "직업계고등학교 실습실 안전관리 간편 매뉴얼.pdf",
    BASE_DIR / "[한국교육시설안전원] 연구활동종사자 실험실습실 안전교육.pdf",
]


def load_saved_api_key() -> str:
    """
    Streamlit Cloud Secrets, .env 또는 api_key.txt 파일에서 Gemini API 키를 안전하게 읽어옵니다.
    """
    # 1. Streamlit Cloud 배포 환경의 Secrets 확인
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
        if "GOOGLE_API_KEY" in st.secrets:
            return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass

    # 2. 로컬 .env 및 api_key.txt 확인
    load_dotenv(BASE_DIR / ".env", override=True)
    if (BASE_DIR / "api_key.txt").exists():
        load_dotenv(BASE_DIR / "api_key.txt", override=True)

    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if key and key != "your_gemini_api_key_here":
        return key

    # 혹시 변수명(GEMINI_API_KEY=) 없이 키(AIza...)만 바로 적어둔 경우 자동 인식
    for fname in [".env", "api_key.txt"]:
        fpath = BASE_DIR / fname
        if fpath.exists():
            for line in fpath.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    _, val = line.split("=", 1)
                    val = val.strip().strip("'\"")
                    if val and val != "your_gemini_api_key_here":
                        return val
                elif line.startswith("AIza"):
                    return line
    return ""


def get_embeddings(api_key: str | None = None) -> GoogleGenerativeAIEmbeddings:
    """Gemini 임베딩 모델 인스턴스를 반환합니다 (강의안 09-11 p.25 기준)."""
    key = (api_key or "").strip() or load_saved_api_key()
    if not key:
        raise ValueError(
            "GEMINI_API_KEY가 설정되지 않았습니다. api_key.txt(.env)에 키를 입력하거나 사이드바에서 입력해주세요."
        )
    os.environ["GOOGLE_API_KEY"] = key
    os.environ["GEMINI_API_KEY"] = key
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=key,
    )


def load_and_enrich_pdfs(pdf_paths: list[Path] | None = None) -> list:
    """
    PyPDFLoader로 PDF를 페이지별로 읽고, 이미지/아웃라인 폰트로 인해 텍스트가 비어 있는 슬라이드는
    보완 텍스트(pdf_ocr_fallback)를 결합하여 Document 리스트를 반환합니다.
    """
    target_paths = pdf_paths or PDF_FILES
    all_docs = []

    for pdf_path in target_paths:
        if not pdf_path.exists():
            print(f"[경고] 파일을 찾을 수 없습니다: {pdf_path}")
            continue

        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        file_name = pdf_path.name

        for doc in docs:
            # PyPDFLoader의 page 메타데이터는 0부터 시작하므로 1을 더해 실제 페이지 번호로 맞춤
            page_num_1based = doc.metadata.get("page", 0) + 1
            doc.metadata["page_number"] = page_num_1based
            doc.metadata["file_name"] = file_name

            raw_text = (doc.page_content or "").strip()

            # 이미지/도형 위주 페이지 보완
            fallback_text = ""
            if "직업계고등학교" in file_name and page_num_1based in HIGHSCHOOL_PDF_FALLBACK:
                fallback_text = HIGHSCHOOL_PDF_FALLBACK[page_num_1based]
            elif "한국교육시설안전원" in file_name and page_num_1based in KOIES_PDF_FALLBACK:
                fallback_text = KOIES_PDF_FALLBACK[page_num_1based]

            if fallback_text:
                if len(raw_text) < 80:
                    doc.page_content = f"[{file_name} - {page_num_1based}쪽]\n{fallback_text}"
                else:
                    doc.page_content = (
                        f"[{file_name} - {page_num_1based}쪽]\n{raw_text}\n\n{fallback_text}"
                    )
            elif raw_text:
                doc.page_content = f"[{file_name} - {page_num_1based}쪽]\n{raw_text}"

            # 의미 있는 텍스트가 있는 페이지만 추가
            if len(doc.page_content.strip()) > 20:
                all_docs.append(doc)

    return all_docs


def build_vector_store(api_key: str | None = None) -> FAISS:
    """
    1) PyPDFLoader로 문서 로드
    2) RecursiveCharacterTextSplitter로 청킹 (chunk_size=1000, chunk_overlap=200)
    3) GoogleGenerativeAIEmbeddings로 임베딩 및 FAISS 벡터 스토어 생성
    4) ./db/faiss 에 로컬 저장
    """
    print("1. PDF 문서를 불러오는 중...")
    docs = load_and_enrich_pdfs()
    print(f"   -> 총 {len(docs)}개 유효 페이지 로드 완료")

    print("2. 텍스트 청킹(RecursiveCharacterTextSplitter) 수행 중...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"   -> 총 {len(chunks)}개 청크(Chunk) 생성 완료")

    print("3. Gemini 임베딩 및 FAISS 벡터 스토어 생성 중...")
    emb = get_embeddings(api_key=api_key)

    # 무료 등급(Free Tier)의 분당 요청 한도(100 RPM)에 걸리지 않도록 배치 단위로 안전하게 임베딩
    batch_size = 40
    vs = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"   -> 임베딩 진행 중: {i + 1}~{min(i + batch_size, len(chunks))} / {len(chunks)}")
        for attempt in range(3):
            try:
                if vs is None:
                    vs = FAISS.from_documents(batch, emb)
                else:
                    vs.add_documents(batch)
                break
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 2:
                    print("   -> 무료 API 분당 한도 대기 중 (35초 후 자동 재시도)...")
                    time.sleep(35)
                else:
                    raise

    FAISS_DB_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(FAISS_DB_DIR))
    print(f"4. FAISS 인덱스 저장 완료: {FAISS_DB_DIR}")
    return vs


if __name__ == "__main__":
    build_vector_store()
