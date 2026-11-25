from pathlib import Path
import streamlit as st

# Import the two high-level pipeline entry points we built in rag_pipeline.py
from rag_pipeline import run_ingestion, ask_question

# Directory where uploaded documents are stored before indexing
DATA_DIR = Path("data/documents")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Configure the browser tab title, icon, and wide layout
st.set_page_config(
    page_title="Enterprise RAG Document Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Enterprise RAG Document Assistant")
st.caption("Ask grounded questions across your internal PDFs and text documents with verifiable citations.")

# --- SIDEBAR: Document Upload & Ingestion Controls ---
with st.sidebar:
    st.header("📂 Knowledge Base")

    # File uploader widget supporting multiple PDFs and TXT files at once
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    # Save any newly uploaded files into data/documents/
    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = DATA_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to '{DATA_DIR}'.")

    # Display all documents currently sitting in data/documents/
    existing_files = [f.name for f in DATA_DIR.iterdir() if f.is_file()]
    st.subheader(f"Indexed Folder ({len(existing_files)} files)")
    if existing_files:
        for fname in existing_files:
            st.markdown(f"- `{fname}`")
    else:
        st.info("No documents uploaded yet.")

    st.divider()

    # Button to trigger offline ingestion (Load -> Split -> Embed -> Qdrant)
    if st.button("🔄 Process & Index Documents", use_container_width=True, type="primary"):
        if not existing_files:
            st.warning("Upload at least one PDF or TXT file first.")
        else:
            with st.spinner("Chunking documents & building Qdrant vector index..."):
                try:
                    total_chunks = run_ingestion(str(DATA_DIR))
                    st.success(f"Done! Indexed {total_chunks} chunks into Qdrant.")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    # Optional button to clear chat history on screen
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- MAIN CHAT AREA ---
# Initialize chat history in Streamlit's session state so messages persist across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render all previous messages in the conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If the assistant message has attached source chunks, show them in an expander
        if msg.get("sources"):
            with st.expander("🔍 View Retrieved Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**{i}. {src['file']} (Page {src['page']})**")
                    st.caption(src["snippet"])

# Chat input box pinned at the bottom of the screen
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    # 1. Display and store the user's question immediately
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Run the RAG pipeline and display the grounded answer + citations
    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating answer..."):
            try:
                result = ask_question(user_query, k=3)
                answer_text = result["answer"]
                raw_sources = result["sources"]

                st.markdown(answer_text)

                # Format source metadata cleanly for UI display and session storage
                formatted_sources = []
                if raw_sources:
                    with st.expander("🔍 View Retrieved Sources"):
                        for i, doc in enumerate(raw_sources, 1):
                            file_name = Path(doc.metadata.get("source", "Unknown")).name
                            page_num = doc.metadata.get("page_label") or (
                                doc.metadata.get("page", 0) + 1
                            )
                            snippet = doc.page_content[:300].replace("\n", " ") + "..."

                            st.markdown(f"**{i}. {file_name} (Page {page_num})**")
                            st.caption(snippet)

                            formatted_sources.append(
                                {
                                    "file": file_name,
                                    "page": page_num,
                                    "snippet": snippet,
                                }
                            )

                # Save assistant response + sources to session history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": formatted_sources,
                    }
                )

            except FileNotFoundError:
                st.warning(
                    "⚠️ Vector index not found! Click **'Process & Index Documents'** in the sidebar first."
                )
            except Exception as e:
                st.error(f"Error generating response: {e}")