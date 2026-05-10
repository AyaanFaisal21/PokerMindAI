FROM python:3.12-slim

WORKDIR /app

COPY services/shared_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ /app/core/
COPY services/strategist/main.py /app/main.py

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
