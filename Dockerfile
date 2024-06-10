# app/Dockerfile

FROM python:3.10-slim

ENV APP_NAME=main.py
ENV PORT_NUMBER=8501

WORKDIR /app

RUN apt-get update
RUN apt-get install -y build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE $PORT_NUMBER

#HEALTHCHECK CMD curl --fail http://localhost:8501/chatmastr/_stcore/health

CMD streamlit run $APP_NAME --server.port=$PORT_NUMBER