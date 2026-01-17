import os

import streamlit as st

from rag_pipeline import (
    MANUAL_PROMPT_TEMPLATE,
    PDF_FOLDER_PATH,
    initialize_rag_pipeline,
)

st.set_page_config(page_title="SmartPDF - RAG (BM25)", page_icon="📄", layout="wide")

with st.sidebar:
    st.title("📁 Gestion des PDF (BM25)")
    os.makedirs(PDF_FOLDER_PATH, exist_ok=True)

    uploaded_files = st.file_uploader(
        "Déposez vos PDF ici", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(PDF_FOLDER_PATH, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("✅ PDF sauvegardés. Clique sur Ré-indexer.")

    st.divider()

    if st.button("🔄 Ré-indexer (BM25)", use_container_width=True):
        st.cache_resource.clear()
        llm, retriever = initialize_rag_pipeline(force_reindex=True)
        if llm and retriever:
            st.toast("Index BM25 mis à jour ✅", icon="✅")
        st.rerun()

st.title("🤖 Chatbot PDF (RAG léger - BM25)")
st.caption("Recherche lexical BM25 (sans embeddings, très léger).")

llm, retriever = initialize_rag_pipeline(force_reindex=False)
if not llm or not retriever:
    st.info("Ajoute un PDF, puis clique sur **Ré-indexer**.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour 🙂 Pose ta question sur tes PDF."}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Pose ta question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche BM25 + génération Gemini..."):
            try:
                hits = retriever(prompt, k=4)
                context = "\n---\n".join([h["text"] for h in hits])

                final_prompt = MANUAL_PROMPT_TEMPLATE.format(
                    context=context, question=prompt
                )
                response = llm.invoke(final_prompt)

                answer = (
                    response.content
                    if not isinstance(response.content, list)
                    else response.content[0].get("text", "")
                )
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

                if hits:
                    with st.expander("🔍 Sources"):
                        for i, h in enumerate(hits):
                            meta = h.get("meta", {})
                            src = os.path.basename(meta.get("source", "Inconnue"))
                            page = meta.get("page", None)
                            page_display = (page + 1) if isinstance(page, int) else "?"
                            st.write(
                                f"**Source {i+1}:** {src} (Page {page_display}) — score {h['score']:.2f}"
                            )
                            st.caption(f"“{h['text'][:220]}…”")

            except Exception as e:
                st.error(f"Erreur : {e}")
