# 📄 SmartPDF-RAG : Chatbot Intelligent avec Gemini 3

Une application de **RAG (Retrieval-Augmented Generation)** moderne et rapide permettant de discuter avec vos documents PDF en utilisant la puissance de **Google Gemini 3**.

## ✨ Points Forts

- **LLM de Pointe** : Propulsé par `gemini-3-flash-preview` pour des réponses instantanées et précises.
- **Gestionnaire Moderne** : Utilise `uv` pour une installation 10x plus rapide et une gestion des dépendances fiable.
- **Indexation Intelligente** : Stockage vectoriel avec **FAISS** permettant la persistance locale des données (évite de re-scanner les PDF à chaque lancement).
- **Interface Intuitive** : Développé avec **Streamlit** pour une expérience de chat fluide.
- **Transparence** : Affichage automatique des sources (extraits de PDF) utilisées pour générer chaque réponse.

## 🛠️ Stack Technique

- **Langage** : Python 3.9+
- **Orchestration** : LangChain
- **IA (LLM & Embeddings)** : Google Generative AI (Gemini 3)
- **Base de Données Vectorielle** : FAISS
- **Gestion de projet** : `uv` & `pyproject.toml`
- **Interface** : Streamlit

## 🚀 Installation Rapide

Ce projet utilise [uv](https://github.com/astral-sh/uv) pour une gestion simplifiée.

### 1. Cloner le projet
```bash
git clone [https://github.com/JulienSchnitzler/SmartPDF_RAG.git](https://github.com/JulienSchnitzler/SmartPDF_RAG.git)
cd SmartPDF_RAG
```

### 2. Initialiser l'environnement
```bash
# Crée le venv et installe toutes les dépendances verrouillées
uv sync
```

### 3. Configurer les secrets
Créez un fichier .env à la racine :
```Plaintext
GOOGLE_API_KEY="VOTRE_CLE_API_GOOGLE"
```
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
├── PDF/                 # Vos documents PDF source
├── faiss_index/         # Index vectoriel généré localement (ignoré par Git)
├── chatbot_app.py       # Interface utilisateur Streamlit
├── rag_pipeline.py      # Cœur du pipeline RAG
├── pyproject.toml       # Configuration et dépendances modernes
└── uv.lock              # Fichier de verrouillage des versions
```

## Développements futurs
à venir ...