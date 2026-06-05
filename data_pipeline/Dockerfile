FROM apache/airflow:2.9.3-python3.11

COPY --chown=airflow:0 requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

ENV PYTHONPATH=/opt/airflow/src:${PYTHONPATH}
