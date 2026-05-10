FROM python:3.12-slim

WORKDIR /app

COPY services/shared_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ /app/core/
COPY services/humanizer/main.py /app/main.py

EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "2"]
