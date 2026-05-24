# ── Stage 1: Build del frontend React ────────────────────────────────────────
FROM node:20-slim AS frontend-builder
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Backend Python con frontend embebido ────────────────────────────
FROM python:3.11-slim

# Dependencias de sistema para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python
# PyTorch CPU-only primero (evita descargar ~2GB de CUDA que no se usa en Docker)
# ultralytics sobreescribe uvicorn — reinstalar uvicorn al final para que prevalezca
COPY requirements.txt ./
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall "uvicorn[standard]>=0.32.0" "fastapi>=0.115.0"

# Copiar código del backend
COPY main.py detector.py ./

# Copiar frontend compilado
COPY --from=frontend-builder /build/dist ./frontend/dist

# Copiar modelo entrenado
# El modelo se pasa via volumen o se incluye en la imagen
# Para incluirlo: descomenta la siguiente línea y pon best.pt en app/
# COPY best.pt ./models/best.pt

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
