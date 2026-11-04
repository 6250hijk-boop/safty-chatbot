@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\streamlit.exe" (
    echo [안내] 가상환경(.venv) 및 패키지를 설치합니다...
    "C:\Users\6250h\.local\bin\uv.exe" venv --python 3.12 .venv
    "C:\Users\6250h\.local\bin\uv.exe" pip install --python .venv\Scripts\python.exe -r requirements.txt
)

echo [실행] 실험·실습실 안전관리 RAG 챗봇(Streamlit)을 시작합니다...
".venv\Scripts\streamlit.exe" run app.py
pause

