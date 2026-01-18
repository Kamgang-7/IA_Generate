---
title: SmartPDF-RAG
emoji: 📄
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: chatbot_app.py
pinned: false
---

# 📄 SmartPDF-RAG : Assistant Intelligent avec Gemini 3 & BM25

Ce projet est une application de RAG (Retrieval-Augmented Generation) permettant d'interroger des documents PDF de manière naturelle. Il utilise la puissance de Google Gemini 3 combinée à un moteur de recherche BM25 pour garantir des réponses précises et sourcées.

## ✨ Points Forts

- **LLM de Pointe** : Propulsé par `gemini-3-flash-preview` pour des réponses instantanées et précises.
- **Recherche de Texte (BM25)** : Utilisation de l'algorithme de classement BM25Okapi pour retrouver les passages les plus pertinents basés sur les termes exacts.
- **Gestionnaire Moderne** : Utilise `uv` pour une installation 10x plus rapide et une gestion des dépendances fiable.
- **Interface Intuitive** : Développé avec **Streamlit** pour une expérience de chat fluide.
- **Transparence** : Affichage automatique des sources (extraits de PDF) utilisées pour générer chaque réponse.
- **Conteneurisation Complète** : Déploiement simplifié via Docker et Docker Compose, incluant un service de linting automatique.

## 🛠️ Stack Technique

- **Orchestration** : LangChain
- **IA (LLM)** : Google Generative AI (Gemini 3)
- **Indexation** : BM25 (via rank-bm25)
- **Gestionnaire de paquets** : uv (Astral) pour des builds ultra-rapides
- **Interface** : Streamlit
- **Qualité du code** : Ruff & Black (via Docker lint)
- **Monitoring** : Langfuse (optionnel)

## 🚀 Installation et Lancement

Ce projet utilise [uv](https://github.com/astral-sh/uv) pour une gestion simplifiée.

1. Prérequis
Créez un fichier .env à la racine du projet :
```bash
GOOGLE_API_KEY="VOTRE_CLE_API_GOOGLE"

# Optionnel (Monitoring)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST="https://cloud.langfuse.com"
```

### 2. Lancement avec Docker (Recommandé)
```bash
docker-compose up --build
```
L'application sera disponible sur http://localhost:8501.

### 3. Installation Locale avec uv
Si vous préférez lancer le projet nativement :
```bash
uv sync
uv run streamlit run chatbot_app.py
```

## 🌍 Déploiement sur Hugging Face Spaces

Ce projet est compatible avec Hugging Face Spaces (SDK Docker).

1. SDK : Streamlit
2. Port : L'application utilise par défaut le port 8501, mais peut être configurée sur 7860 pour HF dans le Dockerfile.
3. Secrets : Ajoutez votre GOOGLE_API_KEY dans les Settings > Variables and secrets de votre Space Hugging Face.

## 📂 Utilisation

1. Placez vos fichiers PDF dans le dossier PDF/.
2. Lancez l'application via uv :
```bash
uv run streamlit run chatbot_app.py
```
3. Posez vos questions ! L'application créera automatiquement un dossier faiss_index/ lors de la première analyse pour accélérer les sessions futures.

## 📁 Structure du projet
```Plaintext
.
├── PDF/                 # Dossier source des documents PDF
├── bm25_index/          # Stockage local de l'index BM25 (manifeste + store)
├── chatbot_app.py       # Interface Streamlit et logique de conversation
├── rag_pipeline.py      # Cœur du pipeline (BM25, Tokenization, LLM)
├── Dockerfile           # Configuration de l'image Docker
├── docker-compose.yml   # Orchestration des services app et lint
└── pyproject.toml       # Dépendances et configuration des outils (Ruff, Black)
```

## 💡 Fonctionnement de l'Indexation

L'application surveille automatiquement le dossier PDF/. Un "fingerprint" (empreinte numérique) est calculé à chaque lancement :

- Si de nouveaux fichiers sont ajoutés ou modifiés, l'index BM25 est reconstruit.
- Sinon, l'index est chargé depuis le disque pour un démarrage instantané.

## Développements futurs
à venir ...