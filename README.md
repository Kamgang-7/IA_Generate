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

![Python](https://img.shields.io/badge/python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B)
![Docker](https://img.shields.io/badge/Docker-2496ED)

Ce projet est une application de RAG (Retrieval-Augmented Generation) permettant d'interroger des documents PDF de manière naturelle. Il utilise la puissance de Google Gemini 3 combinée à un moteur de recherche BM25 pour garantir des réponses précises et sourcées.

## 📌 Sommaire
1. [🚀 Concept](#-concept)
2. [✨ Points forts](#-points-forts)
3. [🛠️ Choix techniques](#️-choix-techniques)
4. [⚠️ Limitations](#️-limitations)
5. [⚙️ Installation et lancement](#️-installation-et-lancement)
6. [🌍 Déploiement sur Hugging Face Spaces](#️-Déploiement-sur-hugging-face-spaces)
7. [📂 Utilisation](#-utilisation)
8. [📁 Structure du projet](#-structure-du-projet)
9. [💡 Fonctionnement de l'indexation](#-fonctionnement-de-lindexation)
10. [Perspectives d'évolution](#-perspectives-dévolution)

## 🚀 Concept
L'application permet d'uploader des documents PDF et de discuter avec eux via une interface de chat. Contrairement à un chatbot classique, celui-ci "lit" vos documents en temps réel pour extraire les passages pertinents avant de générer une réponse, évitant ainsi les hallucinations et garantissant la véracité des informations.

## ✨ Points forts
- **Réponses Sourcées** : Affichage automatique des extraits de PDF utilisés pour chaque réponse.
- **Vitesse & Fiabilité** : Utilisation de `uv` pour des builds ultra-rapides.
- **Architecture Propre** : Code linté (Ruff & Black) et conteneurisé pour un déploiement sans erreurs.
- **Monitoring Intégré** : Suivi des traces et de la latence via Langfuse cloud.

## 🛠️ Choix techniques
Nous avons privilégié des outils offrant un compromis optimal entre simplicité et performance :
- **LLM (IA)** : `gemini-3-flash-preview` pour sa grande fenêtre de contexte et son faible coût.
- **Moteur de recherche (BM25)** : Choisi à la place d'une base de données vectorielle pour sa précision sur les termes techniques exacts et son absence de coût d'embedding.
- **Orchestration** : **LangChain** pour la gestion fluide de la mémoire et du flux RAG.
- **Conteneurisation** : **Docker & Docker Compose** pour garantir un environnement d'exécution identique sur toutes les machines.

## ⚠️ Limitations
- **Format** : Seuls les fichiers `.pdf` sont acceptés pour le moment.
- **Sémantique** : Le moteur BM25 se base sur les mots-clés ; il peut être moins performant qu'un moteur vectoriel sur des questions purement conceptuelles sans termes communs.
- **Stockage** : L'index est stocké localement (`bm25_index/`) et n'est pas persistant sur une base de données cloud.

## 🚀 Installation et lancement

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

### 2. Lancement avec Docker (recommandé)
```bash
docker-compose up --build
```
L'application sera disponible sur http://localhost:8501.

### 3. Installation locale avec uv
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

1. Placez vos fichiers PDF dans le dossier `PDF/`.
2. Lancez l'application via uv :
```bash
uv run streamlit run chatbot_app.py
```
3. Posez vos questions ! L'application créera automatiquement un dossier `bm25_index/` pour stocker les données traitées.

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

## 💡 Fonctionnement de l'indexation

L'application surveille automatiquement le dossier PDF/. Un "fingerprint" (empreinte numérique MD5) est calculé à chaque lancement :

- Un calcul est fait sur l'ensemble des fichiers du dossier `PDF/`.
- Si le fingerprint change (ajout/suppression), l'index se reconstruit automatiquement.
- Sinon, l'index est chargé instantanément depuis le dossier `bm25_index/`.

## Perspective d'évolution
- Intégration d'un mode hybride (BM25 + VectorDB type FAISS).
- Support des fichiers Word/Markdown.
- Gestion de l'OCR pour les PDF scannés.
- intégration d'un dashboard de coût en temps réel (via Langfuse API).