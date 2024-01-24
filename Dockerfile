# app/Dockerfile

FROM python:3.10-alpine3.19

WORKDIR /app

RUN apk update
RUN apk add -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8501

#HEALTHCHECK CMD curl --fail http://localhost:8501/chatmastr/_stcore/health

ENTRYPOINT ["./entrypoint.sh"]