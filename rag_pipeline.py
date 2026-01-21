import hashlib
import json
import os
import re

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# =========================================================
# CONFIGURATION DES CHEMINS ET CONSTANTES
# =========================================================
PDF_FOLDER_PATH = "PDF"
INDEX_PATH = "bm25_index"
MANIFEST_PATH = os.path.join(INDEX_PATH, "manifest.json")
STORE_PATH = os.path.join(INDEX_PATH, "store.json")  # chunks + metas

# Template de réponse pour le LLM (Gemini)
# On lui demande d'être honnête et de citer uniquement le contexte fourni.
MANUAL_PROMPT_TEMPLATE = """
Vous êtes un assistant spécialisé dans la réponse aux questions.
Utilisez uniquement les morceaux de contexte suivants pour répondre à la question.
Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas.
N'essayez pas d'inventer une réponse.
Restez concis, faites des bullet points si nécessaire.

Contexte:
{context}

Question:
{question}

Réponse utile:
"""


# =========================================================
# FONCTIONS UTILITAIRES DE GESTION DE FICHIERS
# =========================================================
def _list_pdfs(folder: str) -> list[str]:
    """Liste tous les fichiers PDF présents dans le dossier cible."""
    if not os.path.exists(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".pdf")])


def _compute_pdf_fingerprint(folder: str) -> str:
    """
    Crée une signature unique (Hash) basée sur les noms, tailles et dates de modification des PDF.
    Cela permet de savoir si l'index doit être mis à jour sans avoir à tout relire.
    """
    pdfs = _list_pdfs(folder)
    h = hashlib.sha256()
    for name in pdfs:
        p = os.path.join(folder, name)
        try:
            stat = os.stat(p)
            h.update(name.encode("utf-8", errors="ignore"))
            h.update(str(stat.st_size).encode())
            h.update(str(int(stat.st_mtime)).encode())
        except FileNotFoundError:
            continue
    return h.hexdigest()


def _read_json(path: str) -> dict:
    """Lecture sécurisée d'un fichier JSON."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_json(path: str, data: dict) -> None:
    """Écriture structurée dans un fichier JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _tokenize(text: str) -> list[str]:
    """
    Nettoie le texte pour le moteur BM25 :
    Met en minuscule et ne garde que les mots alphanumériques (enlève la ponctuation).
    """
    text = text.lower()
    return re.findall(r"[a-zàâçéèêëîïôùûüÿñæœ0-9]+", text)


def _store_exists() -> bool:
    """Vérifie si des données ont déjà été indexées."""
    return os.path.exists(STORE_PATH)


# =========================================================
# INITIALISATION DU PIPELINE RAG
# =========================================================


# @st.cache_resource permet de garder le pipeline en mémoire vive pour ne pas
# le recharger à chaque clic dans Streamlit.
@st.cache_resource(show_spinner=False)
def initialize_rag_pipeline(force_reindex: bool = False):
    load_dotenv()  # Chargement du fichier .env

    # 1. Initialisation du LLM Gemini
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ Clé API GOOGLE_API_KEY introuvable (Secrets HF ou fichier .env).")
        return None, None

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0.1,
            convert_system_message_to_human=True,
        )
    except Exception as e:
        st.error(f"Erreur init Gemini: {e}")
        return None, None

    # 2. Vérification de la nécessité d'indexer
    os.makedirs(PDF_FOLDER_PATH, exist_ok=True)
    os.makedirs(INDEX_PATH, exist_ok=True)

    pdfs = _list_pdfs(PDF_FOLDER_PATH)
    fingerprint = _compute_pdf_fingerprint(PDF_FOLDER_PATH)
    manifest = _read_json(MANIFEST_PATH)
    last_fp = manifest.get("pdf_fingerprint")

    # On indexe si : force_reindex est True, ou si pas d'index, ou si les fichiers ont changé
    should_build = False
    if force_reindex:
        should_build = True
    elif not _store_exists() and pdfs:
        should_build = True
    elif _store_exists() and pdfs and last_fp != fingerprint:
        should_build = True

    try:
        if should_build:
            # --- PHASE D'INDEXATION ---
            st.warning(f"📦 Indexation BM25 en cours : {len(pdfs)} PDF détecté(s)...")
            with st.spinner("Chargement, découpage, indexation BM25..."):

                # Chargement des PDF depuis le dossier
                loader = DirectoryLoader(
                    PDF_FOLDER_PATH,
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader,
                    show_progress=True,
                    use_multithreading=True,
                )
                documents = loader.load()
                if not documents:
                    st.error("Aucun contenu extrait des PDF (PDF vides/protégés ?).")
                    return None, None

                # Découpage du texte en morceaux (chunks) de 1000 caractères
                # L'overlap de 200 permet de garder le contexte entre deux morceaux.
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(documents)

                texts = [c.page_content for c in chunks]
                metas = [c.metadata for c in chunks]

                # Préparation du corpus pour BM25 (tokenization)
                tokenized_corpus = [_tokenize(t) for t in texts]
                bm25 = BM25Okapi(tokenized_corpus)

                # Sauvegarde sur le disque pour les prochaines fois
                _write_json(STORE_PATH, {"texts": texts, "metas": metas})
                _write_json(
                    MANIFEST_PATH,
                    {"pdf_fingerprint": fingerprint, "pdf_count": len(pdfs)},
                )

            st.success("✅ Index BM25 mis à jour !")

        else:
            # Si pas besoin d'indexer et pas de fichiers, on s'arrête là
            if not _store_exists():
                st.info("📄 Ajoute des PDF pour commencer.")
                return None, None

        # --- CHARGEMENT DE L'INDEX EXISTANT ---
        store = _read_json(STORE_PATH)
        texts = store.get("texts", [])
        metas = store.get("metas", [])

        if not texts:
            st.info("📄 Ajoute des PDF pour commencer.")
            return None, None

        # On doit reconstruire l'objet BM25 en mémoire à partir des textes chargés
        tokenized_corpus = [_tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized_corpus)

        # --- DEFINITION DU RETRIEVER ---
        def retriever_fn(query: str, k: int = 4):
            """
            Fonction de recherche :
            Prend une question, la tokenize, et calcule les scores BM25
            par rapport à tous les morceaux de texte.
            """
            q_tok = _tokenize(query)
            scores = bm25.get_scores(q_tok)

            # Récupération des indices des 'k' meilleurs résultats
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            results = []
            for i in top_idx:
                # Score de pertinence BM25
                results.append({"text": texts[i], "meta": metas[i], "score": float(scores[i])})
            return results

        return llm, retriever_fn

    except Exception as e:
        st.error(f"Une erreur majeure est survenue : {e}")
        return None, None
