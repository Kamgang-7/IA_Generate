import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Constantes de Configuration ---

# Dossier contenant vos fichiers PDF
PDF_FOLDER_PATH = "PDF"
# Dossier où sera sauvegardé l'index FAISS
FAISS_INDEX_PATH = "faiss_index"

# Le prompt template que nous utiliserons
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
    
    Cette fonction est mise en cache par Streamlit (@st.cache_resource) pour
    n'être exécutée qu'une seule fois, au démarrage de l'application.
    
    Elle charge les PDF, crée ou charge l'index vectoriel FAISS,
    et configure l'LLM.
    
    Retourne:
        tuple: (llm, retriever) ou (None, None) en cas d'échec.
    """
    
    print("--- INITIALISATION DU PIPELINE RAG (ne devrait s'exécuter qu'une fois) ---")
    load_dotenv()
    
    # 1. Vérifier la clé API
    if "GOOGLE_API_KEY" not in os.environ:
        st.error("Clé API GOOGLE_API_KEY non trouvée. Veuillez l'ajouter à votre fichier .env.")
        return None, None

    # 2. Initialiser l'LLM et les Embeddings
    try:
        # Utilisez le nom de modèle qui fonctionne pour vous
        llm = ChatGoogleGenerativeAI(
                    model="gemini-3-flash-preview", 
                    temperature=0.1, # On baisse un peu la température pour plus de précision
                    convert_system_message_to_human=True 
                    ) #gemini-2.5-flash-lite
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation de Google AI: {e}")
        return None, None

    # 3. Charger ou Créer le Vector Store FAISS
    try:
        # ** NOUVEAU : Assurez-vous que le dossier FAISS existe **
        if not os.path.isdir(FAISS_INDEX_PATH):
            os.makedirs(FAISS_INDEX_PATH, exist_ok=True) # Crée le dossier s'il n'existe pas
            st.info(f"Dossier FAISS créé: {FAISS_INDEX_PATH}")
            
        # Maintenant, vérifier si le fichier d'index existe à l'intérieur du dossier
        index_file_path = os.path.join(FAISS_INDEX_PATH, "index.faiss")

        # Remplacer la condition `if os.path.exists(FAISS_INDEX_PATH):`
        # par `if os.path.exists(index_file_path):`
        if os.path.exists(index_file_path):
            # Charger l'index depuis le disque (rapide)
            with st.spinner("Chargement de l'index vectoriel existant..."):
                # Utilisez `FAISS_INDEX_PATH` comme chemin de dossier
                db = FAISS.load_local(
                    FAISS_INDEX_PATH, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
            st.success("Index vectoriel chargé depuis le disque.")
        else:        
            # Créer l'index depuis les PDF (lent, coûteux en API)
            st.warning(f"Index FAISS non trouvé. Création d'un nouvel index à partir des PDF dans '{PDF_FOLDER_PATH}'...")
            
            if not os.path.isdir(PDF_FOLDER_PATH):
                st.error(f"Dossier PDF non trouvé: {PDF_FOLDER_PATH}. Veuillez le créer.")
                return None, None

            # Charger les documents
            with st.spinner(f"Chargement des documents depuis '{PDF_FOLDER_PATH}'..."):
                directory_loader = DirectoryLoader(
                    PDF_FOLDER_PATH,
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader,
                    show_progress=True,
                    use_multithreading=True
                )
                documents = directory_loader.load()
            
            if not documents:
                st.error(f"Aucun PDF trouvé dans le dossier '{PDF_FOLDER_PATH}'.")
                return None, None
            
            st.info(f"{len(documents)} pages chargées depuis les PDF.")

            # Découper les textes
            with st.spinner("Découpage des documents en chunks..."):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)
            
            st.info(f"{len(texts)} chunks créés.")

            # Créer les embeddings et l'index FAISS
            with st.spinner(f"Création des embeddings et de l'index FAISS... (utilise l'API Google)"):
                db = FAISS.from_documents(texts, embeddings)
            
            # Sauvegarder l'index sur le disque pour la prochaine fois
            with st.spinner("Sauvegarde de l'index sur le disque..."):
                db.save_local(FAISS_INDEX_PATH)
            
            st.success("Nouvel index vectoriel créé et sauvegardé !")

        # 4. Créer le Retriever
        retriever = db.as_retriever()
        print("--- PIPELINE RAG INITIALISÉ AVEC SUCCÈS ---")
        
        return llm, retriever

    except Exception as e:
        st.error(f"Une erreur majeure est survenue lors de la création du pipeline: {e}")
        return None, None