FROM apache/airflow:3.0.2
COPY requirements.txt /requirements.txt

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         git \
         libgomp1 \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
ARG AIRFLOW_VERSION=3.0.2
ARG PYTHON_VERSION=3.12
RUN pip install --no-cache-dir \
  -r /requirements.txt \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"