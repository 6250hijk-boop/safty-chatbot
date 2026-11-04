# 🦺 실험·실습실 안전관리 RAG 챗봇

전주대학교 인공지능학과 **AI융합프로젝트(김영수 교수님)** 강의안 5개 전 과정을 기반으로 구현한 **LangChain + Google Gemini + FAISS + Streamlit** 기반 안전관리 매뉴얼 질의응답 챗봇입니다.

---

## 📚 학습 데이터 (지식베이스 PDF 2종)
1. `직업계고등학교 실습실 안전관리 간편 매뉴얼.pdf` (총 19쪽)
2. `[한국교육시설안전원] 연구활동종사자 실험실습실 안전교육.pdf` (총 112쪽)

> 💡 **이미지/도형 슬라이드 자동 보완 (`pdf_ocr_fallback.py`)**:  
> `PyPDFLoader`가 텍스트를 추출할 때, 그림이나 아웃라인 글꼴로만 제작된 슬라이드 페이지도 누락 없이 검색되도록 구조화 텍스트 보완 모듈을 결합하여 **총 117개 유효 페이지 → 135개 청크(Chunk)**로 색인됩니다.

---

## 🗂️ 파일 구조 및 강의안 매핑

| 파일명 | 설명 | 대응 강의안 |
| :--- | :--- | :--- |
| `.env` / `.gitignore` | `GEMINI_API_KEY` 환경변수 관리 및 보안 설정 | `01-03. RAG 개요·LangChain·개발환경 설치` |
| `ingest.py` | `PyPDFLoader` → `RecursiveCharacterTextSplitter(1000/200)` → `GoogleGenerativeAIEmbeddings` → `FAISS.save_local("./db/faiss")` | `09-11. 문서 로더·텍스트 청커·임베딩`<br>`12-13-08. 벡터 스토어·리트리버` |
| `app.py` | LCEL 체인(`prompt \| llm \| StrOutputParser()`), `MMR` 리트리버, **Pre-Retrieval(Query Rewrite)**, 멀티턴 대화 기억, 출처 페이지 표기, `st.write_stream` 스트리밍 UI | `04-07. LLM 체인·프롬프트·언어모델·출력파서`<br>`12-13-08. Streamlit 챗봇 UI`<br>`AdvancedRAG. 메모리·Pre/Post-Retrieval` |
| `pdf_ocr_fallback.py` | 이미지 위주 PDF 페이지의 텍스트 보완 데이터 | `09-11. 스캔/이미지 페이지 보완` |
| `run_chatbot.bat` | 더블클릭으로 바로 Streamlit 챗봇을 실행하는 배치 스크립트 | 실행 편의 도구 |

---

## 🚀 실행 방법

### 1단계: Gemini API 키 입력
- 폴더 내 `.env` 파일을 열어 발급받은 키를 입력합니다:
  ```env
  GEMINI_API_KEY=AIzaSy...본인의_키_입력
  ```
- *(참고)* `.env`에 입력하지 않아도 **챗봇 실행 후 웹 화면 왼쪽 사이드바**에서 바로 API 키를 입력할 수 있습니다.

### 2단계: 챗봇 실행
- **방법 A (가장 간편)**: 폴더 안의 `run_chatbot.bat` 파일을 더블클릭합니다.
- **방법 B (터미널 명령어)**:
  ```powershell
  cd C:\Users\6250h\Desktop\chatbot
  .\.venv\Scripts\streamlit.exe run app.py
  ```
- 첫 질문 시(또는 사이드바의 **🔄 벡터 스토어 생성 / 재구축** 버튼 클릭 시) 자동으로 `./db/faiss` 폴더에 벡터 인덱스가 생성됩니다.

