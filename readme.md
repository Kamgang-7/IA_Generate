# 📄 PDF-Chat : Chatbot RAG avec Google Gemini

Une application web Streamlit qui permet de "chatter" avec ses propres documents PDF.

Ce projet utilise un pipeline **RAG** (Retrieval-Augmented Generation / Génération Augmentée par la Récupération) pour analyser plusieurs PDF à la fois, en s'appuyant sur le modèle **Gemini de Google** et un "vector store" **FAISS**.

## 🚀 Démo

[Insérez ici une capture d'écran ou un GIF de votre application en action. C'est très important pour un portfolio !]

## ✨ Fonctionnalités

* **Interrogation de PDF Multiples** : Charge et analyse tous les fichiers `.pdf` trouvés dans un dossier `PDF/`.
* **Interface de Chat Intuitive** : Une interface de chatbot simple et réactive construite avec Streamlit.
* **Pipeline RAG** : Utilise un "vector store" FAISS pour trouver les passages les plus pertinents dans vos documents avant de générer une réponse.
* **Propulsé par Gemini** : Utilise les modèles Google Gemini (ex: `gemini-2.5-flash-lite`) pour la compréhension et la génération de réponses.
* **Persistance du "Vector Store"** : L'index FAISS est sauvegardé sur le disque (`faiss_index/`) après le premier traitement. Cela évite d'avoir à retraiter tous les PDF (et de dépenser des crédits API) à chaque redémarrage de l'application.
* **Mise en Cache Intelligente** : Le pipeline RAG est mis en cache (`@st.cache_resource`) pour des réponses instantanées après le chargement initial.
* **(Optionnel) Affichage des Sources** : L'interface peut montrer quels extraits de texte (chunks) ont été utilisés pour formuler la réponse (voir `Améliorations` ci-dessous).

## 🛠️ Stack Technique

* **Frontend** : Streamlit
* **Orchestration RAG** : LangChain
    * `langchain-google-genai` (pour l'LLM et les Embeddings)
    * `langchain-community` (pour les chargeurs de PDF et FAISS)
    * `langchain-text-splitters` (pour le découpage en chunks)
* **Modèle (LLM)** : Google Gemini
* **Embeddings** : Google (`models/embedding-001`)
* **Vector Store** : FAISS (de Meta AI)
* **Utilitaires** : `python-dotenv`, `pypdf`

## ⚙️ Installation et Lancement

Suivez ces étapes pour lancer le projet sur votre machine locale.

### 1. Prérequis

* Python 3.9+
* Un compte Google avec une clé API pour l'API Gemini (disponible sur [Google AI Studio](https://ai.google.dev/)).

### 2. Cloner le Dépôt

```bash
git clone [https://github.com/VOTRE_NOM_UTILISATEUR/VOTRE_NOM_PROJET.git](https://github.com/VOTRE_NOM_UTILISATEUR/VOTRE_NOM_PROJET.git)
cd VOTRE_NOM_PROJET
```

### 3. Installer les Dépendances

Il est fortement recommandé de créer un environnement virtuel :

```bash
# Créer un environnement virtuel
#python -m venv venv
uv venv

# Activer l'environnement
# Sur Windows:
#.\venv\Scripts\activate
.venv\Scripts\activate

# Sur macOS/Linux:
source venv/bin/activate
```

Installez ensuite les bibliothèques Python requises :

```bash
uv pip install -r requirements.txt
```

### 4. Configurer l'Environnement

Créez un fichier `.env` à la racine du projet (vous pouvez copier `.env.example` pour commencer). Ajoutez-y votre clé API Google :

```text
# .env
GOOGLE_API_KEY="VOTRE_CLE_API_SECRETE_ICI"
```

### 5. Ajouter vos PDF

Créez un dossier nommé `PDF` à la racine de votre projet et placez-y tous les documents PDF que vous souhaitez interroger.

```
VOTRE_NOM_PROJET/
├── PDF/
│   ├── document1.pdf
│   └── document2.pdf
├── chatbot_app.py
└── ...
```

### 6. Lancer l'Application

Vous êtes prêt ! Lancez l'application Streamlit :

```bash
streamlit run chatbot_app.py
```

Ouvrez votre navigateur à l'adresse [http://localhost:8501](http://localhost:8501).

La première fois, le traitement des PDF et la création de l'index FAISS peuvent prendre quelques minutes. Les lancements suivants seront quasi-instantanés grâce à la persistance des données.

## 📁 Structure du Projet

```
.
├── PDF/                 # Dossier pour vos fichiers PDF (ignoré par Git)
├── faiss_index/         # Dossier pour l'index FAISS sauvegardé (ignoré par Git)
├── .env                 # Fichier pour les clés API (secret, ignoré par Git)
├── .env.example         # Modèle pour le fichier .env
├── .gitignore           # Fichiers et dossiers à ignorer par Git
├── chatbot_app.py       # Le code principal de l'application Streamlit
├── rag_pipeline.py      # (Optionnel) Logique séparée pour le pipeline RAG
└── requirements.txt     # Dépendances Python
```

## 💡 Améliorations Possibles

* **Afficher les Sources** : Modifier l'interface Streamlit pour afficher les `retrieved_docs` (les chunks de texte source) sous la réponse, afin que l'utilisateur puisse vérifier l'information.
* **Nettoyage de l'Index** : Ajouter un bouton dans Streamlit pour "forcer le re-traitement" des PDF, ce qui supprimerait le dossier `faiss_index/` et reconstruirait la base de données.
* **Support d'autres formats** : Étendre le `DirectoryLoader` pour inclure les fichiers `.txt`, `.docx`, etc.