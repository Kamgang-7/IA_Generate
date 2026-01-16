FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dépendances système utiles (faiss / numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copier le code
COPY . .

# Streamlit
EXPOSE 8501

# Evite les warnings + écoute sur toutes les interfaces
CMD ["streamlit", "run", "chatbot_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
