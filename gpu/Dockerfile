FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir transformers==4.51.3 accelerate==1.6.0

ENV PYTHONUNBUFFERED=1
COPY generate.py .
ENTRYPOINT ["python", "generate.py"]
