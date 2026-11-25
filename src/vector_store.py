from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Import our factory function so ingestion and querying always use the exact same embeddings
from embeddings import get_embedding_model

# Default local folder where Qdrant persists its SQLite/vector index files
VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "enterprise_docs"


def create_vector_store(
    chunks: List[Document],
    persist_dir: str = VECTORSTORE_DIR,
    collection_name: str = COLLECTION_NAME,
) -> QdrantVectorStore:
    """
    Embeds document chunks and stores them in a persistent local Qdrant collection.
    Overwrites any existing collection with the same name to prevent duplicate chunks.
    """
    if not chunks:
        raise ValueError("Cannot create vector store from an empty list of chunks.")

    # Ensure the target folder exists on disk
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    embeddings = get_embedding_model()

    # QdrantVectorStore.from_documents handles:
    # 1. Initializing local disk storage at `path=persist_dir`
    # 2. Creating the collection with 384-dim cosine similarity
    # 3. Generating embeddings in batches and persisting vectors + metadata
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=persist_dir,
        collection_name=collection_name,
        force_recreate=True,  # Prevents duplicate vectors when re-ingesting documents
    )

    print(
        f"Indexed {len(chunks)} chunk(s) into Qdrant collection '{collection_name}' "
        f"at '{persist_dir}/'."
    )
    return vector_store


def load_vector_store(
    persist_dir: str = VECTORSTORE_DIR,
    collection_name: str = COLLECTION_NAME,
) -> QdrantVectorStore:
    """
    Connects to an existing persistent Qdrant collection on disk for fast retrieval.
    """
    embeddings = get_embedding_model()

    # Open the local Qdrant storage directory
    client = QdrantClient(path=persist_dir)

    # Guardrail: fail clearly if the user tries to query before indexing any documents
    if not client.collection_exists(collection_name):
        raise FileNotFoundError(
            f"Collection '{collection_name}' not found in '{persist_dir}/'. "
            "Run ingestion first to index your documents."
        )

    # Wrap the existing client in LangChain's QdrantVectorStore interface
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    return vector_store


if __name__ == "__main__":
    # End-to-end ingestion test: Load -> Split -> Embed & Index -> Similarity Search
    from document_loader import load_documents
    from text_splitter import split_documents

    raw_docs = load_documents()
    chunks = split_documents(raw_docs)

    # Build and persist the vector index
    db = create_vector_store(chunks)

    # Test semantic retrieval right away (no LLM call needed yet!)
    test_query = "What is the main purpose of the AI Interview Copilot?"
    results = db.similarity_search(test_query, k=2)

    print(f"\nTop {len(results)} retrieved chunk(s) for query: '{test_query}'")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} (Page {doc.metadata.get('page_label', 'N/A')}) ---")
        print(doc.page_content[:250])