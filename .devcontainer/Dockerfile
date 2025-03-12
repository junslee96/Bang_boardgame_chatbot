FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye

# 필요한 패키지 설치
RUN apt update && apt install -y git curl

# pip를 사용하여 Python 패키지 설치
COPY requirements.txt /tmp/
RUN pip3 install --user -r /tmp/requirements.txt

# Tornado 버전 조정
RUN pip3 install --user tornado==5.0

# nest_asyncio 설치
RUN pip3 install --user nest_asyncio

# streamlit 설치
RUN pip3 install --user streamlit

# 작업 디렉토리 설정
WORKDIR /workspaces

# 명령어 실행
CMD ["streamlit", "run", "streamlit_app.py", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
