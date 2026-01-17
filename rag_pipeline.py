import os
import json
import hashlib
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

PDF_FOLDER_PATH = "PDF"
INDEX_PATH = "bm25_index"
MANIFEST_PATH = os.path.join(INDEX_PATH, "manifest.json")
STORE_PATH = os.path.join(INDEX_PATH, "store.json")  # chunks + metas

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


def _list_pdfs(folder: str) -> list[str]:
    if not os.path.exists(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".pdf")])


def _compute_pdf_fingerprint(folder: str) -> str:
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
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _tokenize(text: str) -> list[str]:
    # Tokenizer simple, léger, robuste
    text = text.lower()
    return re.findall(r"[a-zàâçéèêëîïôùûüÿñæœ0-9]+", text)


def _store_exists() -> bool:
    return os.path.exists(STORE_PATH)


@st.cache_resource
def initialize_rag_pipeline(force_reindex: bool = False):
    load_dotenv()

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

    os.makedirs(PDF_FOLDER_PATH, exist_ok=True)
    os.makedirs(INDEX_PATH, exist_ok=True)

    pdfs = _list_pdfs(PDF_FOLDER_PATH)
    fingerprint = _compute_pdf_fingerprint(PDF_FOLDER_PATH)
    manifest = _read_json(MANIFEST_PATH)
    last_fp = manifest.get("pdf_fingerprint")

    should_build = False
    if force_reindex:
        should_build = True
    elif not _store_exists() and pdfs:
        should_build = True
    elif _store_exists() and pdfs and last_fp != fingerprint:
        should_build = True

    try:
        if should_build:
            st.warning(f"📦 Indexation BM25 en cours : {len(pdfs)} PDF détecté(s)...")
            with st.spinner("Chargement, découpage, indexation BM25..."):
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

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                chunks = splitter.split_documents(documents)

                texts = [c.page_content for c in chunks]
                metas = [c.metadata for c in chunks]

                tokenized_corpus = [_tokenize(t) for t in texts]
                bm25 = BM25Okapi(tokenized_corpus)

                # On sauvegarde les chunks + metas, et on reconstruit bm25 au chargement (rapide)
                _write_json(STORE_PATH, {"texts": texts, "metas": metas})
                _write_json(
                    MANIFEST_PATH,
                    {"pdf_fingerprint": fingerprint, "pdf_count": len(pdfs)},
                )

            st.success("✅ Index BM25 mis à jour !")

        else:
            if not _store_exists():
                st.info("📄 Ajoute des PDF pour commencer.")
                return None, None

        # Charger store + reconstruire BM25 (léger et rapide)
        store = _read_json(STORE_PATH)
        texts = store.get("texts", [])
        metas = store.get("metas", [])
        if not texts:
            st.info("📄 Ajoute des PDF pour commencer.")
            return None, None

        tokenized_corpus = [_tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized_corpus)

        # retriever BM25 maison
        def retriever_fn(query: str, k: int = 4):
            q_tok = _tokenize(query)
            scores = bm25.get_scores(q_tok)
            # top-k indices
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
                :k
            ]
            results = []
            for i in top_idx:
                results.append(
                    {"text": texts[i], "meta": metas[i], "score": float(scores[i])}
                )
            return results

        return llm, retriever_fn

    except Exception as e:
        st.error(f"Une erreur majeure est survenue : {e}")
        return None, None
