FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY sandbox_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY sandbox_service/main.py /app/main.py
COPY pipeline/barrios/data/barris-barrios.geojson /data/valencia_barrios.geojson

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
