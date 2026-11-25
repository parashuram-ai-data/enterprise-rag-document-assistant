# 📄 Enterprise RAG Document Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** application that lets users upload internal PDF and TXT documents, index them into a local persistent vector database, and ask grounded questions with page-level citations.

## 🏗️ Architecture & Tech Stack
* **Orchestration:** LangChain (`langchain`, `langchain-core`, `langchain-text-splitters`)
* **Embeddings (Local & Free):** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384-dim L2-normalized vectors)
* **Vector Store:** Qdrant (embedded persistent local disk storage with cosine similarity)
* **LLM Generation:** Google Gemini (`gemini-3.8-flash` with automatic retry & fallback to `gemini-3.6-flash` / `gemini-3.5-flash-lite`)
* **Frontend UI:** Streamlit with multi-file upload, one-click re-indexing, and expandable source citations

## 📁 Project Structure
```text
enterprise-rag-document-assistant/
├── data/
│   └── documents/          # Raw PDF and TXT source files
├── src/
│   ├── __init__.py         # Package initializer
│   ├── document_loader.py  # Multi-format ingestion (PDF, TXT) with metadata preservation
│   ├── text_splitter.py    # Recursive chunking (chunk_size=1000, overlap=200)
│   ├── embeddings.py       # Local HuggingFace embedding model factory
│   ├── vector_store.py     # Persistent Qdrant collection creation & retrieval
│   ├── rag_pipeline.py     # Grounded prompt assembly, retry/fallback LLM calls & citations
│   └── app.py              # Interactive Streamlit chat & document management UI
├── vectorstore/            # Local persistent Qdrant index (ignored in Git)
├── .env.example            # Environment variable template
├── .gitignore              # Git exclusions (venv, .env, vectorstore index)
└── requirements.txt        # Python dependencies