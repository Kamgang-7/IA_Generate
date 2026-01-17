import os
import json
import hashlib
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

PDF_FOLDER_PATH = "PDF"
FAISS_INDEX_PATH = "faiss_index"
MANIFEST_PATH = os.path.join(FAISS_INDEX_PATH, "manifest.json")

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


def _read_manifest() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        return {}
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_manifest(data: dict) -> None:
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _index_exists() -> bool:
    return os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss"))


@st.cache_resource
def initialize_rag_pipeline(force_reindex: bool = False):
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ GOOGLE_API_KEY introuvable (Secrets HF ou fichier .env).")
        return None, None

    # LLM Gemini (uniquement pour répondre, pas pour embeddings)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
        convert_system_message_to_human=True,
    )

    # Embeddings légers (ONNX) — pas de torch
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"  # petit et performant (anglais)
    )

    os.makedirs(PDF_FOLDER_PATH, exist_ok=True)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

    pdfs = _list_pdfs(PDF_FOLDER_PATH)
    fingerprint = _compute_pdf_fingerprint(PDF_FOLDER_PATH)
    manifest = _read_manifest()
    last_fp = manifest.get("pdf_fingerprint")

    should_build = force_reindex or (
        pdfs and (not _index_exists() or last_fp != fingerprint)
    )

    try:
        if should_build:
            if not pdfs:
                st.info("📄 Ajoute des PDF pour commencer.")
                return None, None

            st.warning(f"📦 Indexation embeddings (FastEmbed) : {len(pdfs)} PDF...")
            loader = DirectoryLoader(
                PDF_FOLDER_PATH,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
                use_multithreading=True,
            )
            documents = loader.load()
            if not documents:
                st.error("Aucun contenu extrait des PDF.")
                return None, None

            # Réduit la taille de l'index (moins de chunks = moins d'espace)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200, chunk_overlap=150
            )
            chunks = splitter.split_documents(documents)

            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(FAISS_INDEX_PATH)

            _write_manifest(
                {
                    "pdf_fingerprint": fingerprint,
                    "pdf_count": len(pdfs),
                    "chunk_count": len(chunks),
                }
            )
            st.success("✅ Index FAISS mis à jour !")

        db = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={"k": 4})
        return llm, retriever

    except Exception as e:
        st.error(f"Erreur: {e}")
        return None, None
