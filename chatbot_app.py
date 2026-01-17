import os
import streamlit as st
from rag_pipeline import (
    initialize_rag_pipeline,
    MANUAL_PROMPT_TEMPLATE,
    PDF_FOLDER_PATH,
)

st.set_page_config(
    page_title="SmartPDF - RAG (Embeddings légers)", page_icon="📄", layout="wide"
)

with st.sidebar:
    st.title("📁 PDF")
    os.makedirs(PDF_FOLDER_PATH, exist_ok=True)

    files = st.file_uploader("Dépose tes PDF", type="pdf", accept_multiple_files=True)
    if files:
        for f in files:
            with open(os.path.join(PDF_FOLDER_PATH, f.name), "wb") as out:
                out.write(f.getbuffer())
        st.success("✅ PDF sauvegardés. Clique sur Ré-indexer.")

    if st.button("🔄 Ré-indexer", use_container_width=True):
        st.cache_resource.clear()
        initialize_rag_pipeline(force_reindex=True)
        st.rerun()

st.title("🤖 Chatbot PDF (Embeddings légers)")
llm, retriever = initialize_rag_pipeline(force_reindex=False)
if not llm or not retriever:
    st.info("Ajoute un PDF puis clique sur Ré-indexer.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour 🙂 Pose ta question."}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("Ta question..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Recherche + réponse..."):
            docs = retriever.invoke(q)
            context = "\n---\n".join([d.page_content for d in docs])

            prompt = MANUAL_PROMPT_TEMPLATE.format(context=context, question=q)
            r = llm.invoke(prompt)
            ans = (
                r.content
                if not isinstance(r.content, list)
                else r.content[0].get("text", "")
            )
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
