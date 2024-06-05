# app/Dockerfile

FROM python:3.10-slim

ENV app_name=main.py
ENV port_num=8501

WORKDIR /app

RUN apt-get update
RUN apt-get install -y build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8501

#HEALTHCHECK CMD curl --fail http://localhost:8501/chatmastr/_stcore/health

ENTRYPOINT ["./entrypoint.sh"]
#ENTRYPOINT ["streamlit run ${app_name} --server.port=${port_num}"]