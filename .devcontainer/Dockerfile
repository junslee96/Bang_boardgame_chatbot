FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye

# 필요한 시스템 패키지 설치 (Java 포함)
RUN apt update && apt install -y git curl openjdk-11-jdk-headless

# JAVA_HOME 환경 변수 설정
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64
ENV PATH "$PATH:$JAVA_HOME/bin"

# pip 사용하여 Python 패키지 설치
COPY requirements.txt /tmp/
RUN pip3 install --user -r /tmp/requirements.txt

# Tornado 및 nest_asyncio 설치
RUN pip3 install --user tornado==6.2 nest-asyncio

# 작업 디렉토리 설정
WORKDIR /workspaces

# Streamlit 실행 명령어 설정
CMD ["streamlit", "run", "streamlit_app.py", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
