FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye

# 필요한 시스템 패키지 설치 (Java 제거)
RUN apt update && apt install -y git curl

# pip 사용하여 Python 패키지 설치
COPY requirements.txt /tmp/
RUN pip3 install --user -r /tmp/requirements.txt

# 작업 디렉토리 설정
WORKDIR /workspaces

# Streamlit 실행 명령어 설정
CMD ["streamlit", "run", "streamlit_app.py", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
