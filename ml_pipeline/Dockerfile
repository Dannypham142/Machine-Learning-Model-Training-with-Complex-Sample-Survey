FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_ALLOW_FILE_STORE=true

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data_processing.py logistic_regression.py neural_network.py xgb.py pums_labels.py ./

EXPOSE 5001

CMD ["bash"]
