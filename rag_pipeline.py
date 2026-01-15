import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Constantes de Configuration ---
PDF_FOLDER_PATH = "PDF"
FAISS_INDEX_PATH = "faiss_index"

MANUAL_PROMPT_TEMPLATE = """
Vous êtes un assistant spécialisé dans la réponse aux questions.
Utilisez uniquement les morceaux de contexte suivants pour répondre à la question.
Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas.
N'essayez pas d'inventer une réponse.
Restez concis, faites des bullet points si nécessaire.
Soyez poli et professionnel.

Contexte:
{context}

Question:
{question}

Réponse utile:
"""

# --- Logique Principale du Pipeline RAG ---

@st.cache_resource
def initialize_rag_pipeline():
    """
    Initialise l'ensemble du pipeline RAG.
    """
    
    print("--- INITIALISATION DU PIPELINE RAG ---")
    load_dotenv()
    
    # 1. Vérifier la clé API
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ Clé API non trouvée ! Vérifiez vos Secrets Hugging Face ou votre fichier .env.")
        return None, None

    # 2. Initialiser l'LLM et les Embeddings
    try:
        llm = ChatGoogleGenerativeAI(
                    model="gemini-3-flash-preview", 
                    temperature=0.1,
                    convert_system_message_to_human=True 
                    )
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation de Google AI: {e}")
        return None, None

    # 3. Charger ou Créer le Vector Store FAISS
    try:
        if not os.path.isdir(FAISS_INDEX_PATH):
            os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
            
        index_file_path = os.path.join(FAISS_INDEX_PATH, "index.faiss")

        # --- MODIFICATION ICI : On vérifie d'abord si on a des PDF à traiter ---
        # Cela permet au bouton "Ré-indexer" de prendre en compte les nouveaux fichiers
        pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.lower().endswith('.pdf')] if os.path.exists(PDF_FOLDER_PATH) else []

        if pdf_files:
            # Créer l'index depuis les PDF (priorité pour permettre la mise à jour)
            st.warning(f"Analyse de {len(pdf_files)} PDF dans '{PDF_FOLDER_PATH}'...")
            
            with st.spinner("Chargement et découpage des documents..."):
                directory_loader = DirectoryLoader(
                    PDF_FOLDER_PATH,
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader,
                    show_progress=True,
                    use_multithreading=True
                )
                documents = directory_loader.load()
                
                if not documents:
                    st.error(f"Aucun contenu extrait des PDF dans '{PDF_FOLDER_PATH}'.")
                    return None, None

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)
                
                # Création des embeddings et de l'index
                db = FAISS.from_documents(texts, embeddings)
                
                # Sauvegarde (écrase l'ancien index)
                db.save_local(FAISS_INDEX_PATH)
                st.success("Base de connaissances mise à jour avec succès !")
        
        elif os.path.exists(index_file_path):
            # Si pas de PDF mais un index existe (chargement simple)
            with st.spinner("Chargement de l'index existant..."):
                db = FAISS.load_local(
                    FAISS_INDEX_PATH, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
            st.info("Utilisation de l'index vectoriel existant.")
        else:
            # Ni PDF ni Index
            st.info("En attente de documents PDF...")
            return None, None

        # 4. Créer le Retriever
        retriever = db.as_retriever()
        print("--- PIPELINE RAG INITIALISÉ AVEC SUCCÈS ---")
        return llm, retriever

    except Exception as e:
        st.error(f"Une erreur majeure est survenue : {e}")
        return None, None