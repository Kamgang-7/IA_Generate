import streamlit as st
import os
# On importe tout le nécessaire depuis votre pipeline
from rag_pipeline import initialize_rag_pipeline, MANUAL_PROMPT_TEMPLATE, PDF_FOLDER_PATH

# --- Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="SmartPDF - Gemini 3 RAG",
    page_icon="📄",
    layout="wide"
)

# --- Barre Latérale : Gestion des Documents ---
with st.sidebar:
    st.title("📁 Gestion des PDF")
    st.write("Ajoutez vos documents pour alimenter l'IA.")

    # 1. Création du dossier PDF s'il n'existe pas
    if not os.path.exists(PDF_FOLDER_PATH):
        os.makedirs(PDF_FOLDER_PATH)

    # 2. Zone d'upload
    uploaded_files = st.file_uploader(
        "Déposez vos PDF ici", 
        type="pdf", 
        accept_multiple_files=True
    )

    # 3. Traitement des fichiers uploadés
    if uploaded_files:
        files_saved = False
        for uploaded_file in uploaded_files:
            file_path = os.path.join(PDF_FOLDER_PATH, uploaded_file.name)
            # On écrit le fichier s'il n'existe pas encore
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                files_saved = True
        
        if files_saved:
            st.success("Nouveaux documents détectés !")

        # --- Bouton de ré-indexation avec retour visuel ---
        if st.button("🔄 Lancer la ré-indexation"):
            with st.status("Mise à jour de la base de connaissances...", expanded=True) as status:
                st.write("Nettoyage du cache système...")
                st.cache_resource.clear()
                
                st.write("Analyse des PDF et création des embeddings...")
                # On force l'initialisation pour reconstruire l'index FAISS
                llm, retriever = initialize_rag_pipeline()
                
                status.update(label="Indexation terminée avec succès !", state="complete", expanded=False)
            
            st.toast("L'IA est à jour !", icon="✅")
            st.rerun()

    st.divider()
    st.caption("Propulsé par Gemini 3 Flash & FAISS")

# --- Corps Principal ---
st.title("🤖 Chatbot pour vos PDF 📄")
st.caption("Posez des questions sur vos documents en temps réel.")

# --- Initialisation du Pipeline RAG ---
try:
    # Cette fonction est cachée, elle ne recalculera rien sauf si on a vidé le cache
    llm, retriever = initialize_rag_pipeline()
except Exception as e:
    st.error(f"Erreur lors du démarrage : {e}")
    llm, retriever = None, None

# Gestion du cas où aucun document n'est présent
if not llm or not retriever:
    st.info("👋 **Bienvenue !** Pour commencer, veuillez ajouter un ou plusieurs fichiers PDF dans la barre latérale à gauche.")
    st.stop()

# --- Initialisation de l'historique du Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Bonjour ! J'ai analysé vos documents. Comment puis-je vous aider ?"
    }]

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Logique de Question/Réponse ---
if prompt := st.chat_input("Posez votre question ici..."):
    
    # 1. Message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Réponse assistant
    with st.chat_message("assistant"):
        with st.spinner("Recherche et réflexion..."):
            try:
                # Recherche des documents pertinents
                retrieved_docs = retriever.invoke(prompt)
                
                # Formatage du contexte
                context_string = "\n---\n".join([doc.page_content for doc in retrieved_docs])
                
                # Préparation du prompt final
                final_prompt = MANUAL_PROMPT_TEMPLATE.format(
                    context=context_string,
                    question=prompt
                )
                
                # Appel à Gemini 3
                response = llm.invoke(final_prompt)

                # Extraction du texte (Gestion spécifique Gemini 3)
                if isinstance(response.content, list):
                    answer = response.content[0].get('text', '')
                else:
                    answer = response.content

                # Affichage du résultat
                st.markdown(answer)
                
                # Affichage des sources
                if retrieved_docs:
                    with st.expander("🔍 Voir les sources consultées"):
                        for i, doc in enumerate(retrieved_docs):
                            source_file = doc.metadata.get('source', 'Inconnue')
                            source_page = doc.metadata.get('page', 'Inconnue')
                            # Nettoyage du nom de fichier pour l'affichage
                            file_name = os.path.basename(source_file)
                            st.write(f"**Source {i+1} :** {file_name} (Page {source_page+1})")
                            st.caption(f'"{doc.page_content[:200]}..."')

                # Sauvegarde dans l'historique
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")