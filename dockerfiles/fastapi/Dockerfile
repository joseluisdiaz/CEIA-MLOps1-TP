FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /requirements.txt
COPY . .

RUN apt-get update && apt-get install -y curl git libgomp1
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

EXPOSE 8800
