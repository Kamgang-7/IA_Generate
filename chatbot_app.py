import streamlit as st
from rag_pipeline import initialize_rag_pipeline, MANUAL_PROMPT_TEMPLATE

# --- Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="Chat avec vos PDF",
    page_icon="📄",
    layout="centered"
)

st.title("Chatbot pour vos PDF 📄")
st.caption("Posez des questions sur n'importe quel document de votre dossier 'PDF'")

# --- Initialisation du Pipeline RAG ---
# Cela utilise le cache : le code dans `initialize_rag_pipeline` 
# ne s'exécute qu'une fois.
try:
    llm, retriever = initialize_rag_pipeline()
except Exception as e:
    st.error(f"Une erreur est survenue lors du démarrage : {e}")
    llm, retriever = None, None

# Si l'initialisation échoue, on arrête l'application
if not llm or not retriever:
    st.warning("Le pipeline RAG n'a pas pu être initialisé. Vérifiez les erreurs ci-dessus et votre fichier .env.")
    st.stop()

# --- Initialisation de l'historique du Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Bonjour ! Je suis prêt à répondre à vos questions sur les documents du dossier PDF."
    }]

# Afficher les messages de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Logique du Chat ---
if prompt := st.chat_input("Posez votre question ici..."):
    
    # 1. Ajouter le message de l'utilisateur à l'historique et l'afficher
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Préparer et afficher la réponse de l'assistant
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les documents..."):
            try:
                # 3. Récupérer les documents pertinents (le "R" de RAG)
                retrieved_docs = retriever.invoke(prompt)
                
                # 4. Formater le contexte
                context_string = "\n---\n".join([doc.page_content for doc in retrieved_docs])
                
                # 5. Formater le prompt final
                final_prompt = MANUAL_PROMPT_TEMPLATE.format(
                    context=context_string,
                    question=prompt
                )
                
                # --- 6. Générer la réponse (le "G" de RAG) ---
                st.spinner("Génération de la réponse...")
                response = llm.invoke(final_prompt)

                # --- NOUVELLE LOGIQUE D'EXTRACTION ---
                # On vérifie si le contenu est une liste (cas du modèle Gemini 3 Preview) 
                # ou une simple chaîne de caractères.
                if isinstance(response.content, list):
                    # On extrait le texte du premier élément de la liste
                    answer = response.content[0].get('text', '')
                else:
                    # Cas classique
                    answer = response.content

                # On ignore volontairement les 'extras' ou les signatures pour l'affichage
                # ---------------------------------------

                # 7. Afficher la réponse
                st.markdown(answer)
                
                # 8. (Amélioration) Afficher les sources utilisées
                with st.expander("Afficher les sources"):
                    for i, doc in enumerate(retrieved_docs):
                        source_file = doc.metadata.get('source', 'Inconnue')
                        source_page = doc.metadata.get('page', 'Inconnue')
                        st.write(f"**Source {i+1} (Fichier: {source_file}, Page: {source_page+1})**")
                        st.caption(f'"{doc.page_content[:250]}..."')

                # 9. Ajouter la réponse de l'assistant à l'historique
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Une erreur est survenue lors de la génération de la réponse : {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Désolé, une erreur est survenue: {e}"})