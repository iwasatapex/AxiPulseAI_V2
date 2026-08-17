FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1         PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN test -f models/operation_health_predictor.joblib         && test -f models/nps_predictor_model.pkl

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
