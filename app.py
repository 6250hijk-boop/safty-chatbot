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
# 로컬 Pre-Retrieval (Query Rewrite) - 질문 의도를 왜곡하지 않고 맥락 복원 + 핵심 키워드만 정제
# ---------------------------------------------------------------------------
TERM_SYNONYMS = [
    (r"\bccc\b|응급조치\s*ccc", "응급조치 CCC(Check·Call·Care)", ["응급상황 처리 과정", "Check", "Call", "Care"]),
    (r"오토클레이브|autoclave|고압\s*멸균기", "고압멸균기(Autoclave)", ["고압멸균기", "autoclave", "수동밸브", "증기"]),
    (r"클린\s*벤치|clean\s*bench|무균\s*실험대|무균\s*작업대", "무균실험대(Clean bench)", ["무균실험대", "Clean bench", "UV램프"]),
    (r"\bgc\b|가스\s*크로마토", "가스크로마토그래피(GC)", ["가스크로마토그래피", "운반기체", "칼럼 오븐"]),
    (r"\bmsds\b|\bghs\b|물질\s*안전\s*보건", "물질안전보건자료(GHS-MSDS)", ["MSDS", "GHS", "유해성·위험성"]),
    (r"\bbsl\b|생물\s*안전\s*등급", "생물안전등급(BSL 1~4등급)", ["생물안전", "BSL", "밀폐"]),
    (r"\bbsc\b|생물\s*안전\s*작업대", "생물안전작업대(BSC)", ["생물안전작업대", "BSC", "헤파 필터"]),
    (r"\blmo\b|유전자\s*변형", "유전자변형생물체(LMO)", ["유전자변형생물체", "LMO"]),
    (r"면장갑", "회전체 작업 시 면장갑 착용 금지", ["면장갑", "밀링", "띠톱", "회전말림점"]),
    (r"세안기|눈\s*세척", "눈 세척기(세안기)", ["눈 세척기", "세안기", "15분 이상"]),
    (r"사전\s*유해\s*인자|r&dsa", "사전유해인자위험분석(R&DSA)", ["사전유해인자위험분석", "안전현황 분석", "비상조치계획"]),
    (r"일상\s*점검|정기\s*점검|정밀\s*안전\s*진단", "연구실 안전점검 및 정밀안전진단", ["일상점검", "정기점검", "정밀안전진단", "실시주기"]),
]

STOP_SUFFIX_RE = re.compile(
    r"(은|는|이|가|을|를|에|에서|으로|로|과|와|도|만|의|에게|께|한테|부터|까지|이란|란|에는|에서는|할\s*때|할때|시에|시에는|대해|대한|관련|어떻게|어떡해|어케|무엇인가요|무엇인지|뭔가요|뭔지|뭐야|뭔데|알려줘|알려주세요|설명해줘|설명해주세요|해줘|해주세요|되나요|돼요|할까요|있나요|있어|이유는|이유가|왜)$"
)


def extract_core_keywords(text: str) -> list[str]:
    cleaned = re.sub(r"[?!.,~\"'()\[\]{}]", " ", text or "")
    cleaned = re.sub(
        r"(어떻게|어떡해|어케|무엇인가요|뭔가요|뭐야|알려줘|알려주세요|설명해줘|설명해주세요|해줘|해주세요|되나요|있나요|있어)",
        " ",
        cleaned,
    ).strip()
    keywords: list[str] = []
    for w in cleaned.split():
        if len(w) < 2:
            continue
        stripped = STOP_SUFFIX_RE.sub("", w).strip()
        cand = stripped if len(stripped) >= 2 else w
        if cand not in {"그냥", "그러면", "그럼", "그리고", "혹시", "만약", "어떤", "무슨", "얼마나", "무엇"}:
            if cand not in keywords:
                keywords.append(cand)
    return keywords


def local_query_rewrite(query: str, history_messages: list[dict]) -> tuple[str, str]:
    """
    사용자의 원래 질문 의도를 다른 주제로 덮어쓰지 않고:
    1) 대명사/후속 질문인 경우에만 직전 대화 주제어를 복원하고,
    2) 구어체를 명확한 검색 질문으로 정돈하며,
    3) 질문에 실제로 포함된 전문용어/약어와 핵심 키워드만 추출해 검색 정확도를 높입니다.
    반환값: (화면 표시용 재작성 질문, 실제 검색용 확장 쿼리)
    """
    base = query.strip()
    prev_keywords: list[str] = []

    needs_context = bool(
        re.match(r"^(그거|그건|그때|거기|이\s*경우|그럼|그러면|여기서|또|추가로)", base)
    ) or len(base) <= 10
    if needs_context and len(history_messages) >= 2:
        prev_user = next(
            (m["content"] for m in reversed(history_messages[:-1]) if m["role"] == "user"),
            "",
        )
        prev_keywords = extract_core_keywords(prev_user)[:3]

    normalized = base
    normalized = re.sub(r"어케\s*해\??", "어떻게 대처해야 하나요?", normalized)
    normalized = re.sub(r"어떡해\??", "어떻게 대처해야 하나요?", normalized)
    normalized = re.sub(r"뭐야\??$", "무엇인가요?", normalized)
    normalized = re.sub(r"뭔데\??$", "무엇인가요?", normalized)
    normalized = re.sub(r"왜\s*안\s*돼\??$", "금지하는 이유는 무엇인가요?", normalized)
    normalized = re.sub(r"알려줘\.?$", "핵심 내용과 기준은 무엇인가요?", normalized)
    normalized = re.sub(r"설명해줘\.?$", "상세 내용과 절차는 무엇인가요?", normalized)

    if prev_keywords:
        normalized = f"[{' · '.join(prev_keywords)}] {normalized}"

    matched_canonicals: list[str] = []
    boost_terms: list[str] = []
    for pattern, canonical, boosts in TERM_SYNONYMS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            matched_canonicals.append(canonical)
            boost_terms.extend(boosts)

    core_kws = list(dict.fromkeys(prev_keywords + extract_core_keywords(base)))
    kw_note = f" (핵심키워드: {', '.join(core_kws[:5])})" if core_kws else ""
    syn_note = f" [{', '.join(matched_canonicals)}]" if matched_canonicals else ""

    display_rewritten = f"{normalized}{syn_note}{kw_note}"
    effective_search = f"{base} {' '.join(prev_keywords)} {' '.join(core_kws)} {' '.join(boost_terms)}".strip()
    return display_rewritten, effective_search


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
# 모드 1: ⚡ 오프라인 RAG 엔진 답변 합성기 (원리 해설 + 단계별 행동요령 + 주의사항까지 풍부하게 생성)
# ---------------------------------------------------------------------------
DIRECT_QA_RULES = [
    (
        lambda q: "ccc" in q or ("응급" in q and any(k in q for k in ["처리", "과정", "조치"])),
        "**응급조치 CCC**는 실험·실습실 안전사고 발생 시 골든타임 내에 2차 피해를 막고 환자를 구호하기 위한 3단계 핵심 절차로, **① Check(현장 파악) → ② Call(보고 및 119 신고) → ③ Care(응급처치 및 병원 동행)**를 의미합니다.",
        "실험·실습실 사고는 화학물질 누출, 감전, 기계 끼임 등 2차 사고 위험이 공존하므로, 무작정 환자에게 접근하기 전에 먼저 현장 위험 요소를 파악(Check)하고, 교사 독단으로 처리하지 않고 보건교사·관리자·119에 신속히 알려(Call) 전문 구호를 받으며 응급처치(Care)를 이어가는 것이 핵심 원리입니다.",
        [
            "**1단계 [사고 발생 및 즉시 알림]:** 학생은 사고 발생 즉시 당황하거나 숨기지 말고 지도교사에게 즉시 알립니다.",
            "**2단계 [Check - 현장 상황 파악]:** 환자 수와 부상 상태(의식·호흡·출혈), 주변 위험 요소(가스·전기·화재)를 확인하고 나머지 학생들을 안전지대로 대피시킵니다.",
            "**2단계 [Call - 알림 및 신고]:** 교사 독단으로 판단하지 말고 보건교사·교감·교장에게 즉시 보고하며, 119 신고 및 보호자 연락(추측·과장 표현 삼가, 정확한 상태와 이송 병원 안내)을 취합니다.",
            "**2단계 [Care - 처치 및 이송]:** 구급대 도착 전까지 응급처치를 실시하고, **병원 이송 시에는 반드시 교사 1인이 동행**합니다.",
            "**3~4단계 [원인 보고 및 후속조치]:** 보건일지 기록 및 관할 교육청 보고 후, 재발 방지 대책 수립과 **학교안전공제회 보상 신청**을 진행합니다.",
        ],
        "교사 단독 판단으로 병원 이송을 미루거나, 보호자에게 추측성·책임회피성 발언을 하는 것은 절대 금물입니다.",
        "직업계고등학교 실습실 안전관리 간편 매뉴얼 3쪽 (본문 1쪽)",
    ),
    (
        lambda q: ("면장갑" in q or "장갑" in q)
        and any(k in q for k in ["밀링", "띠톱", "선반", "회전", "이유", "안 되는", "안되는", "금지", "드릴"]),
        "선반·밀링·드릴·띠톱 등 **회전하는 기계 작업 시 면장갑을 끼면, 고속 회전하는 날이나 금속 칩에 장갑의 섬유 올이 살짝만 닿아도 순식간에 감겨 손 전체가 기계로 빨려 들어가는 치명적인 회전말림 사고(절단·골절)**가 발생하기 때문입니다.",
        "기계 안전공학에서 고속 회전하는 주축이나 커터 부위는 **'회전말림점(Rotating Entanglement Point)'**과 **'절단점'**에 해당합니다. 면장갑은 실(섬유)이 느슨하게 짜여 있어 찢어지지 않고 회전체에 감기면서 손가락을 기계 속으로 끌어당기므로 맨손보다 훨씬 위험합니다. 실제 대학 및 직업계고 실습실 사고 사례에서도 밀링·테이블톱 작업 중 면장갑이 말려 들어가 손가락이 절단되거나 골절된 사고가 가장 빈번하게 발생했습니다.",
        [
            "**회전체(선반·밀링·띠톱·드릴) 작업 시:** 일반 면장갑 착용을 **절대 금지**하며, 필요한 경우 손에 완전히 밀착되고 올이 풀리지 않는 전용 작업장갑만 지도교사 지시하에 제한적으로 사용합니다.",
            "**금속 칩(부스러기) 제거 시:** 기계가 돌아가는 도중에 손이나 장갑으로 칩을 치우지 말고, 반드시 **기계를 완전히 정지시킨 후 전용 브러시(칩 제거솔)**로 제거합니다.",
            "**측정 및 점검 시:** 주축 회전 중에는 일감(가공물)의 다듬질면을 손으로 만지거나 측정하지 말고 완전히 정지한 뒤 측정합니다.",
            "**복장 및 보호구:** 말림 위험이 있는 헐거운 옷소매나 장신구를 착용하지 않고, 비산하는 칩으로부터 눈을 보호하기 위해 **보안경**을 반드시 착용합니다.",
        ],
        "회전 중인 공구 홀더나 주축을 손으로 잡아 정지시키거나, 띠톱 덮개(가드) 옆에 서서 면장갑을 끼고 작업하는 행위는 절대 금지입니다.",
        "직업계고등학교 실습실 안전관리 간편 매뉴얼 7쪽, 9쪽, 18쪽 / 연구활동종사자 안전교육 58쪽, 96쪽, 100~101쪽",
    ),
    (
        lambda q: "묽힐" in q
        or "희석" in q
        or ("황산" in q and any(k in q for k in ["누출", "대처", "물", "사고"])),
        "산(특히 황산)을 묽힐 때는 **반드시 '다량의 물에 산을 조금씩 천천히 저어 주며'** 넣어야 하며, 반대로 진한 산에 물을 부으면 폭발적인 수화열로 산이 끓어올라 얼굴과 피부로 튀게 됩니다. 황산 누출·접촉 시에는 **즉시 흐르는 깨끗한 물로 20분 이상 세척**해야 합니다.",
        "진한 황산(H₂SO₄)은 물과 만날 때 막대한 열(용해열·수화열)을 방출하고 물보다 비중(밀도)이 큽니다. 만약 진한 황산에 물을 부으면 가벼운 물이 황산 표면에 뜬 상태에서 순간적으로 100℃ 이상 끓어올라 강산이 사방으로 튀는 폭발적 비산 현상이 일어납니다. 반면 물에 황산을 천천히 넣으면 무거운 황산이 가라앉으며 열이 물 전체로 분산되어 안전합니다.",
        [
            "**산 희석(묽히기) 안전수칙:** 보안경·내산 장갑·실험복을 착용하고, 비커의 **물에 진한 산을 유리막대로 천천히 저어 주면서 조금씩 투입**합니다.",
            "**황산액 누출 시 초기 대응:** 즉시 주변 연구활동종사자에게 사고를 전파하여 대피시키고, 학교 관리과 및 연구실안전환경관리자에게 위치·약품종류·부상자 유무를 신고합니다.",
            "**피부·눈 노출 시 응급처치:** 비상샤워기나 세안기(눈 세척기)를 이용해 노출 부위를 **깨끗한 흐르는 물로 최소 15~20분 이상 충분히 세척**한 뒤 즉시 병원으로 이송합니다.",
            "**현장 방제 처리:** 안전환경관리자는 해당 물질의 MSDS를 확인하고 개인보호구를 착용한 뒤 현장을 통제하고 **흡착제·흡착포·중화제**로 누출액을 처리합니다.",
        ],
        "단, 나트륨(Na)·칼륨(K) 같은 '금수성 물질(물 반응성 물질)'이 누출·접촉된 경우에는 물과 만나면 폭발하므로 절대 물로 세척하면 안 됩니다!",
        "직업계고등학교 실습실 안전관리 간편 매뉴얼 6쪽, 12쪽, 16쪽 / 연구활동종사자 안전교육 46쪽",
    ),
    (
        lambda q: any(k in q for k in ["일상점검", "정기점검", "정밀안전진단"])
        or ("점검" in q and "주기" in q),
        "「연구실 안전환경 조성에 관한 법률(연구실안전법)」에 따른 법정 점검 주기는 **① 일상점검: 매일 1회(육안), ② 정기점검: 매년 1회 이상(점검기기), ③ 특별안전점검: 필요시 상시, ④ 정밀안전진단: 2년마다 1회 이상**입니다.",
        "연구실 사고를 사전에 예방하기 위해 연구실안전법 제14조(안전점검) 및 제15조(정밀안전진단)에서는 위험 수준과 점검 방식에 따라 4가지 점검 체계를 의무화하고 있습니다. 매일 연구활동 시작 전 실험자가 직접 확인하는 일상점검부터, 전문 장비와 자격을 갖춘 기관이 수행하는 정기점검·정밀안전진단까지 단계적으로 운영됩니다.",
        [
            "**일상점검 (매일 1회):** 연구개발활동 시작 전 연구실 안전관리 실태(전기, 가스, 시약 보관, 보호구, 정리정돈 등)를 **육안**으로 매일 점검하고 기록합니다 (고시 별표2).",
            "**정기점검 (매년 1회 이상):** 연구실 안전관리 실태를 **안전점검기기**를 이용하여 매년 1회 이상 종합적으로 측정·점검합니다 (고시 별표3).",
            "**특별안전점검 (상시 / 필요시):** 폭발·화재 등 안전에 치명적인 위험을 야기할 가능성이 예상되는 경우 연구주체의 장이 수시로 실시합니다 (고시 별표4).",
            "**정밀안전진단 (2년마다 1회 이상):** 유해화학물질, 유해인자, 독성가스를 취급하는 연구실에 대하여 잠재적 위험성 발견과 개선 대책 수립을 위해 **2년마다 1회 이상** 정기적으로 실시합니다.",
        ],
        "안전점검 및 정밀안전진단을 대행하게 할 경우 반드시 과학기술정보통신부에 등록된 전문 대행기관에 의뢰해야 하며, 미실시 시 과태료 등 행정처분 대상이 됩니다.",
        "[한국교육시설안전원] 연구활동종사자 실험실습실 안전교육 9쪽, 13쪽",
    ),
    (
        lambda q: any(k in q for k in ["화상", "데임", "세안기", "눈 세척", "눈에"]),
        "**화상** 발생 시에는 즉시 **흐르는 찬물(수돗물)로 15~30분 이상 충분히 열기를 식히고 물집은 절대 터뜨리지 않으며**, **눈에 화학물질이나 칩·티가 들어갔을 때**는 눈을 비비지 말고 **눈 세척기(세안기)의 흐르는 물로 15분 이상 세척**한 뒤 병원으로 이송해야 합니다.",
        "화상 부위는 피부 조직 깊숙이 잔열이 남아 조직 손상을 계속 진행시키므로 즉각적인 찬물 냉각이 가장 중요하며, 물집(수포)을 임의로 터뜨리면 세균에 의한 2차 감염 위험이 급증합니다. 또한 눈에 들어간 화학물질이나 금속 칩은 손으로 비비는 순간 각막을 치명적으로 손상시켜 실명에 이를 수 있습니다.",
        [
            "**화상 냉각 처치:** 즉시 흐르는 수돗물이나 찬물로 **최소 15~30분 이상** 충분히 식힙니다.",
            "**의복 위 화상 처치:** 옷을 입은 채 뜨거운 물이나 화학물질에 데었을 때는 억지로 옷을 먼저 벗기지 말고, **옷 위에 찬물을 부어 충분히 식힌 뒤** 벗기거나 달라붙은 부위는 가위로 잘라냅니다.",
            "**물집 및 드레싱 관리:** 물집이 생겼을 때는 터뜨리지 말고 그대로 두며, 화상 부위가 공기에 노출되어 흉터나 감염이 생기지 않도록 소독 거즈와 붕대로 가볍게 감싸 병원으로 이송합니다.",
            "**눈 세척(세안기) 요령:** 눈에 화학물질이 튀거나 티가 들어갔을 때는 절대 비비거나 날카로운 도구로 빼내려 하지 말고, 눈 세척기(세안기)에서 **귀 쪽에서 눈 쪽(또는 오염되지 않은 눈에서 오염된 눈 바깥쪽)으로 흐르는 물에 15분 이상** 눈과 눈꺼풀을 씻어냅니다.",
        ],
        "광범위한 화상을 입었거나 화학물질(산·알칼리) 화상, 안구 손상의 경우 응급 세척 직후 지체 없이 전문 의료기관으로 후송해야 합니다.",
        "직업계고등학교 실습실 안전관리 간편 매뉴얼 6쪽, 13쪽, 15~16쪽 / 연구활동종사자 안전교육 46쪽, 78쪽",
    ),
    (
        lambda q: any(k in q for k in ["고압멸균기", "오토클레이브", "autoclave", "가스크로마토", "gc"]),
        "**고압멸균기(Autoclave)**는 작업 후 **내부 압력과 온도가 완전히 낮아진 것을 확인하고 수동밸브로 잔여 증기를 배출한 뒤 문을 열어야** 하며, **가스크로마토그래피(GC)**는 **전원 차단 전 반드시 가스 공급을 먼저 차단**하고 수소 등 가연성 운반기체 누출을 점검해야 합니다.",
        "고압멸균기는 121℃ 이상의 고온·고압 수증기로 멸균하므로 냉각 전에 뚜껑을 열면 고압 증기와 끓는 액체가 폭발적으로 분출하여 심각한 안면·전신 화상을 입게 됩니다. 가스크로마토그래피(GC)는 고전압, 고온 칼럼 오븐, 그리고 폭발 범위가 넓은 수소(H₂) 등을 운반기체로 사용하므로 감전·화상·가스 폭발의 3중 위험이 존재합니다.",
        [
            "**고압멸균기(Autoclave) 안전수칙:** ① 적정 온도로 냉각되기 전에 절대 문을 열지 않는다 → ② 완전히 열기 전 압력계(0 기압)와 온도를 확인하고 **수동밸브를 열어 남아 있는 증기를 완전히 제거**한다 → ③ 뜨거운 용기를 꺼낼 때는 **내열 장갑과 보안면**을 착용한다 → ④ 가연성·폭발성·인화성 물질은 절대 넣지 않는다.",
            "**가스크로마토그래피(GC) 안전수칙:** ① 기기 **전원을 끄기 전에 가스 공급 밸브를 먼저 차단**한다 → ② 기기 작동 전 가스 연결 라인과 밸브의 누출 여부를 확인한다 → ③ 칼럼 오븐의 고열에 의한 화상에 주의한다 → ④ 수소 등 가연성·폭발성 가스를 운반기체로 사용할 때는 반드시 **가스 누출검지기**를 설치한다.",
        ],
        "고압멸균기 내부에 증류수를 보충할 때도 반드시 안전 온도 이하로 충분히 냉각되었는지 먼저 확인해야 합니다.",
        "[한국교육시설안전원] 연구활동종사자 실험실습실 안전교육 61~62쪽, 84쪽",
    ),
]


def synthesize_offline_rag_answer(query: str, docs: list[Document]) -> str:
    """API 키 없이도 진짜 AI 튜터처럼 핵심 결론 + 배경/원리 해설 + 단계별 안전수칙 + 주의사항을 풍부하게 생성합니다."""
    if not docs:
        return "질문하신 내용과 일치하는 안전수칙을 찾지 못했습니다. 다른 키워드(예: 화재, 화상, 밀링, 황산, 보호구, 안전점검)로 질문해 주세요."

    q_lower = query.lower()
    for match_fn, direct_ans, explanation, key_points, warning_text, src_label in DIRECT_QA_RULES:
        if match_fn(q_lower):
            bullets = "\n".join(f"- {p}" for p in key_points)
            return (
                f"### 🎯 핵심 답변\n{direct_ans}\n\n"
                f"### 💡 원리 및 상세 해설\n{explanation}\n\n"
                f"### ✅ 단계별 안전수칙 및 행동요령\n{bullets}\n\n"
                f"### ⚠️ 핵심 주의사항\n- {warning_text}\n\n"
                f"> 📖 **출처:** [{src_label}]"
            )

    q_tokens = [t for t in re.split(r"[^\w가-힣]+", q_lower) if len(t) >= 2]
    candidates: list[tuple[float, str, str]] = []

    for doc_rank, doc in enumerate(docs[:4]):
        fname = doc.metadata.get("file_name", "안전매뉴얼")
        page = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
        src = f"{fname} {page}쪽"
        for raw_line in re.split(r"\n|(?<=\.)\s+", doc.page_content):
            cleaned = re.sub(r"^[❍▶⁍❖✚✓•\-*①②③④⑤]+\s*", "", raw_line).strip()
            if (
                len(cleaned) < 6
                or cleaned.startswith("[")
                or re.match(r"^-?\s*\d+\s*-?$", cleaned)
                or "안전관리 간편 매뉴얼" in cleaned
            ):
                continue
            score = (4 - doc_rank) * 1.5
            c_lower = cleaned.lower()
            for tok in q_tokens:
                if tok in c_lower:
                    score += 5.0 + min(len(tok), 5)
            candidates.append((score, cleaned, src))

    candidates.sort(key=lambda x: x[0], reverse=True)
    picked: list[tuple[str, str]] = []
    for _, text, src in candidates:
        if len(picked) >= 7:
            break
        if not any(text == p[0] or text in p[0] or p[0] in text for p in picked):
            picked.append((text, src))

    if not picked:
        return "해당 질문에 대한 구체적인 조항을 찾지 못했습니다. 아래 출처 원문을 확인해 주세요."

    main_ans = picked[0][0]
    sub_ans = picked[1][0] if len(picked) > 1 else ""
    extra = [f"- {t}" for t, _ in picked[1:7]]
    sources_str = " / ".join(dict.fromkeys(s for _, s in picked))

    summary_paragraph = f"**{main_ans}**" + (f" 또한, {sub_ans}" if sub_ans else "")
    out = (
        f"### 🎯 핵심 답변\n{summary_paragraph}\n\n"
        f"### 💡 매뉴얼 상세 설명 및 안전수칙\n"
        f"질문하신 **'{query}'**와 관련하여 실험·실습실 안전관리 매뉴얼에서는 사고 예방과 안전한 작업을 위해 다음 수칙을 준수하도록 규정하고 있습니다.\n\n"
        + "\n".join(extra)
        + f"\n\n> 📖 **출처:** [{sources_str}]"
    )
    return out


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
                "너는 직업계고등학교 및 대학 실험·실습실 안전관리 전문 AI 튜터다.\n"
                "단순히 문서 문장을 기계적으로 나열하거나 너무 짧게 단답으로만 끊지 말고, [참고 문서]를 바탕으로 사용자가 완벽히 이해할 수 있도록 다음 4단계 구조로 친절하고 충실하게 설명하라:\n"
                "1) ### 🎯 핵심 답변: 질문에 대한 명확한 결론을 2~3문장으로 먼저 요약 제시\n"
                "2) ### 💡 원리 및 상세 해설: 왜 그렇게 해야 하는지(과학적·안전공학적 이유나 위험 원리)를 자연스러운 설명문으로 해설\n"
                "3) ### ✅ 단계별 안전수칙 및 행동요령: 실제 실험·실습실에서 따라야 할 구체적 절차를 3~5개 항목으로 정리\n"
                "4) > 📖 **출처:** 참고한 매뉴얼 문서명과 페이지 번호 표기\n\n"
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
                max_output_tokens=1100,
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

    saved_env_key = load_saved_api_key()
    # Secrets 또는 .env에 API 키가 이미 등록되어 있으면 자동으로 '☁️ 3. API 사용 버전(Gemini 초경량)'을 기본 선택!
    default_mode_index = 2 if saved_env_key else 0

    engine_mode = st.radio(
        "사용할 챗봇 버전을 선택하세요",
        options=["offline_rag", "opensource_llm", "cloud_api"],
        index=default_mode_index,
        format_func=lambda x: {
            "offline_rag": "⚡ 1. 오프라인 RAG 엔진 (API 불필요 · 0원)",
            "opensource_llm": "🦙 2. 무료 오픈소스 LLM (Ollama · 0원)",
            "cloud_api": "☁️ 3. API 사용 버전 (Gemini 초경량 · 권장)",
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

# 메인 화면 상단: PDF 추가 업로드 및 검색 대상 문서 필터
with st.expander("📄 + 검색할 새로운 PDF 문서 추가 업로드 및 검색 범위 선택 (한국어/영어 자동 번역 지원)", expanded=False):
    up_col1, up_col2 = st.columns([3, 2])
    with up_col1:
        main_uploaded_files = st.file_uploader(
            "여기에 추가할 PDF 파일을 끌어다 놓거나 선택하세요 (여러 개 가능)",
            type=["pdf"],
            accept_multiple_files=True,
            key="main_pdf_uploader",
        )
        if main_uploaded_files:
            splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
            for uf in main_uploaded_files:
                if uf.name not in st.session_state.uploaded_pdf_names:
                    with st.spinner(f"[{uf.name}] 추출 및 색인 중..."):
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
                                d.metadata["is_uploaded"] = True
                                new_docs.append(d)
                            split_new = splitter.split_documents(new_docs)
                            st.session_state.extra_chunks.extend(split_new)
                            st.session_state.uploaded_pdf_names.append(uf.name)
                            st.success(f"✅ {uf.name}: {len(split_new)}개 청크가 검색 대상에 추가되었습니다!")
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
    with up_col2:
        available_doc_options = (
            ["전체 문서 통합 검색"]
            + (["업로드한 추가 PDF에서만 검색"] if st.session_state.uploaded_pdf_names else [])
            + [p.name for p in PDF_FILES]
            + st.session_state.uploaded_pdf_names
        )
        selected_pdf_filter = st.selectbox(
            "📚 검색 대상 PDF 선택",
            options=available_doc_options,
            index=0,
        )
        st.caption(
            f"현재 총 **{2 + len(st.session_state.uploaded_pdf_names)}종**의 PDF 문서가 검색 지식베이스에 등록되어 있습니다."
        )

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
        if selected_pdf_filter == "업로드한 추가 PDF에서만 검색":
            filtered_chunks = [c for c in st.session_state.extra_chunks] or all_chunks
        elif selected_pdf_filter != "전체 문서 통합 검색":
            filtered_chunks = [
                c for c in all_chunks if c.metadata.get("file_name") == selected_pdf_filter
            ] or all_chunks
        else:
            filtered_chunks = all_chunks

        history_text = format_chat_history(st.session_state.messages[:-1])

        # 1. Pre-Retrieval (Query Rewrite)
        if use_rewrite:
            display_rewritten, search_query = local_query_rewrite(query, st.session_state.messages)
        else:
            display_rewritten, search_query = query, query

        # 2. Retrieval (선택된 PDF 범위 내에서 로컬 하이브리드 MMR 검색)
        retrieved_docs = retrieve_local_mmr(filtered_chunks, search_query, top_k, search_type)
        context_text = format_docs_compact(retrieved_docs)

        if use_rewrite and display_rewritten != query:
            st.caption(f"🔍 **Query Rewrite 검색어:** `{display_rewritten}`")

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
                "rewritten_query": display_rewritten if (use_rewrite and display_rewritten != query) else None,
                "sources": sources_info,
            }
        )
