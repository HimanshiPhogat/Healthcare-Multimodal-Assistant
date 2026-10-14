FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "mmhealth.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
