# Official LangChain v1+ wrapper for HuggingFace sentence-transformer models
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> HuggingFaceEmbeddings:
    """
    Initializes and returns a local HuggingFace embedding model.

    Args:
        model_name: HuggingFace repository ID for the sentence-transformer model.
                    'all-MiniLM-L6-v2' outputs 384-dimensional vectors and runs
                    fast on CPU with strong semantic retrieval accuracy.
    """
    # Force CPU execution so the code runs reliably on any laptop without CUDA errors
    model_kwargs = {"device": "cpu"}

    # Normalize vectors to unit length (L2 norm = 1) so cosine similarity
    # comparisons in the vector database are mathematically consistent
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embeddings


if __name__ == "__main__":
    # Quick smoke test: embed a single sample query and inspect its vector dimensions
    embed_model = get_embedding_model()
    sample_vector = embed_model.embed_query("What is this document about?")

    print(f"Embedding model loaded successfully!")
    print(f"Vector dimension: {len(sample_vector)}")
    print(f"First 5 values: {sample_vector[:5]}")