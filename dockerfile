FROM python:3.11-slim
WORKDIR /app

# python:3.11-slim no trae curl ni wget, y el healthcheck de Coolify necesita
# uno de los dos para consultar el endpoint desde dentro del contenedor.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Precarga los pesos en el cache de la imagen. Sin esto el contenedor baja
# 255 MB en cada arranque y el healthcheck lo puede matar antes de responder.
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; \
    AutoTokenizer.from_pretrained('LordMathi2741/distilbert-es'); \
    M.from_pretrained('LordMathi2741/distilbert-es')"

EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "presentation.app:app"]