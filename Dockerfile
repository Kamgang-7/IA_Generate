FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installation directe depuis pyproject.toml
COPY pyproject.toml .
RUN uv pip install --system --no-cache .

# Copie du reste du projet [cite: 3]
COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "chatbot_app.py", "--server.address=0.0.0.0", "--server.port=7860"]