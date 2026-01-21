import os

import streamlit as st

# =========================================================
# 1. INTEGRATION LANGFUSE (Monitoring & Tracing)
# =========================================================
# On tente d'importer le CallbackHandler pour LangChain.
# Si le module n'est pas installé, l'app ne crash pas grâce au try/except.
try:
    from langfuse.langchain import CallbackHandler

    HAS_LANGFUSE = True
except Exception:
    HAS_LANGFUSE = False

from rag_pipeline import (
    MANUAL_PROMPT_TEMPLATE,
    PDF_FOLDER_PATH,
    initialize_rag_pipeline,
)

# Configuration de l'interface Streamlit (titre de l'onglet, icône, mode large
st.set_page_config(page_title="SmartPDF - Assistant RAG Intelligent", page_icon="🚀", layout="wide")

# =========================================================
# 2. TEMPLATES DE PROMPTS
# =========================================================
# Ce prompt sert à "contextualiser" la question.
REWRITE_PROMPT = """
Sur la base de l'historique de la conversation et de la dernière
question de l'utilisateur, reformule une question autonome qui 
peut être comprise sans l'historique. 
Cette question servira à faire une recherche dans des documents PDF.

Historique :
{history}

Dernière question : {question}

Question reformulée (sois précis et direct) :
"""

# =========================================================
# 3. CONFIGURATION LANGFUSE & CALLBACKS
# =========================================================
pk = os.getenv("LANGFUSE_PUBLIC_KEY")
sk = os.getenv("LANGFUSE_SECRET_KEY")

langfuse_handler = None
if HAS_LANGFUSE and pk and sk:
    langfuse_handler = CallbackHandler()
    st.sidebar.success("✅ Langfuse activé (LLM tracing)")
else:
    st.sidebar.warning("ℹ️ Langfuse désactivé (keys absentes ou module manquant)")

st.sidebar.divider()

# Le dictionnaire 'config' sera passé aux appels LLM pour envoyer les logs à Langfuse
config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}

# =========================================================
# 4. GESTION DE L'ETAT (Session State)
# =========================================================
# Streamlit recharge tout le script à chaque interaction.
# On utilise st.session_state pour garder les données en mémoire.

# Initialisation de l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Flag pour savoir si l'index BM25 doit être reconstruit (ex: après un upload)
if "index_needed" not in st.session_state:
    st.session_state.index_needed = False


# =========================================================
# 5. FONCTIONS D'AIDE (Helpers)
# =========================================================
def _extract_text(content):
    """Nettoie la sortie du LLM pour s'assurer qu'on récupère bien une chaîne de caractères."""
    if isinstance(content, list) and len(content) > 0:
        first = content[0]
        if isinstance(first, dict):
            return first.get("text", str(first))
        return str(first)
    return str(content)


def build_index_now():
    """Déclenche la création de l'index BM25 à partir des fichiers PDF du dossier source."""
    st.cache_resource.clear()
    initialize_rag_pipeline(force_reindex=True)
    st.session_state.index_needed = False


# =========================================================
# 6. BARRE LATÉRALE (Sidebar)
# =========================================================
with st.sidebar:
    st.header("1) Charger des documents PDF")
    os.makedirs(PDF_FOLDER_PATH, exist_ok=True)  # Crée le dossier PDF s'il n'existe pas

    # Widget de téléchargement multiple
    uploaded_files = st.file_uploader("Chargez un ou plusieurs PDFs", type="pdf", accept_multiple_files=True)

    # Sauvegarde physique des fichiers sur le serveur/PC
    if uploaded_files:
        files_saved = False
        for f in uploaded_files:
            file_path = os.path.join(PDF_FOLDER_PATH, f.name)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as file:
                    file.write(f.getbuffer())
                files_saved = True

        if files_saved:
            st.session_state.index_needed = True
            st.session_state.last_upload_msg = f"{len(uploaded_files)} fichier(s) prêt(s) à l'indexation."

    # Affichage persistant du message de succès
    if st.session_state.index_needed and "last_upload_msg" in st.session_state:
        st.success(st.session_state.last_upload_msg)
        st.warning("⚠️ Cliquez sur le bouton ci-dessous pour mettre à jour l'IA.")

    st.divider()

    st.header("2) Recréer l'index")
    if st.button("🔄 Re-générer l'index", use_container_width=True):
        with st.spinner("Indexation en cours..."):
            build_index_now()
            # On nettoie le message après indexation
            if "last_upload_msg" in st.session_state:
                del st.session_state.last_upload_msg
        st.success("✅ Index mis à jour !")
        st.rerun()

    st.divider()

    # Information pédagogique sur les scores de pertinence
    st.info(
        """
        **💡 Score BM25 (Confiance) :**
        * **> 10** : Très pertinent ✅
        * **< 2** : Peu précis / Aléatoire ⚠️
        
        *Plus le score est élevé, plus la source est fiable.*
        """
    )

# =========================================================
# 7. CORPS DE L'APPLICATION
# =========================================================
st.title("SmartPDF - Assistant RAG Intelligent 🤖")

# Guide rapide pour l'utilisateur
with st.expander("Guide de démarrage rapide", expanded=True):
    st.markdown(
        """
    Bienvenue sur **SmartPDF** ! Pour poser des questions à vos documents, suivez ces étapes :
    1.  **Charger vos documents** : Utilisez le bouton dans la barre latérale pour uploader vos PDF.
    2.  **Indexer les fichiers** : Cliquez sur **'Re-générer l'index'**.
    3.  **Discutez** : Posez votre question dans la barre de chat en bas de l'écran. 
    
    *Note : Si vous oubliez d'indexer, le système le fera automatiquement lors de votre première question.*
    """
    )

# Initialisation silencieuse du pipeline (charge l'index existant si disponible)
llm, retriever = initialize_rag_pipeline()

# Affichage de tous les messages précédents (historique de session)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Zone de saisie utilisateur
prompt = st.chat_input("Votre question (vous pouvez écrire même si l'index n'est pas encore prêt).")

if prompt:
    # On ajoute la question de l'utilisateur à l'historique et on l'affiche
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse de l'assistant
    with st.chat_message("assistant"):

        # --- ETAPE 1 : Verification de l'Index ---
        # 1) Si aucun index n'existe ou si de nouveaux fichiers attendent, on indexe avant de répondre
        if (not callable(retriever)) or st.session_state.index_needed:
            with st.spinner("📦 Construction de l’index BM25 (une seule fois)..."):
                build_index_now()
                # Recharger pipeline après indexation
                llm, retriever = initialize_rag_pipeline()

        # 2) Sécurité si le chargement échoue
        if not llm or not callable(retriever):
            st.error("Pipeline non prêt (vérifie tes PDFs / GOOGLE_API_KEY).")
            st.stop()

        # --- ETAPE 2 : Reformulation (Query Rewriting) ---
        search_query = prompt
        history_str = ""
        # Si on a déjà discuté, on demande au LLM de créer une question autonome
        if len(st.session_state.messages) > 1:
            past_messages = st.session_state.messages[-4:-1]
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in past_messages])

            rewrite_input = REWRITE_PROMPT.format(history=history_str, question=prompt)
            rewrite_res = llm.invoke(rewrite_input, config=config)
            search_query = _extract_text(rewrite_res.content)

        st.caption(f"🔍 **Requête optimisée :** *{search_query}*")

        # --- ETAPE 3 : Recherche (Retrieval) ---
        with st.spinner("Recherche dans les documents..."):
            hits = retriever(search_query, k=4)
            doc_context = "\n---\n".join([h["text"] for h in hits])

        # --- ETAPE 4 : Génération de la réponse (Generation) ---
        with st.spinner("Rédaction de la réponse..."):
            combined_context = f"[HISTORIQUE]\n{history_str}\n\n[DOCUMENTS]\n{doc_context}"
            final_prompt = MANUAL_PROMPT_TEMPLATE.format(context=combined_context, question=prompt)

            # Appel final au LLM (Gemini)
            response = llm.invoke(final_prompt, config=config)
            answer = _extract_text(response.content)

            st.markdown(answer)
            # Sauvegarde de la réponse dans l'historique
            st.session_state.messages.append({"role": "assistant", "content": answer})

        # --- ETAPE 5 : Affichage des Sources ---
        if hits:
            with st.expander("🔍 Sources consultées & Pertinence"):
                for h in hits:
                    score = h.get("score", 0)
                    source_name = os.path.basename(h["meta"]["source"])
                    page_num = h["meta"]["page"] + 1

                    st.write(f"📄 **{source_name}** (Page {page_num})")
                    st.code(f"Score BM25 : {score:.2f}", language="markdown")
                    st.text(f"“{h['text'][:200]}…”")
                    st.divider()
