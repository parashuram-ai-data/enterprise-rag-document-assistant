from typing import List
from langchain_core.documents import Document
# Dedicated standalone package for text splitting in LangChain v1+
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Splits full-page Document objects into smaller overlapping chunks
    while preserving original metadata (source filename and page number).

    Args:
        documents: List of loaded LangChain Document objects.
        chunk_size: Maximum character length per chunk.
        chunk_overlap: Number of overlapping characters shared between adjacent chunks.
    """
    if not documents:
        print("[WARN] No documents provided to split.")
        return []

    # RecursiveCharacterTextSplitter tries to split by paragraphs first ("\n\n"),
    # then lines ("\n"), then words (" "), keeping semantic units together.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    # split_documents() automatically copies each parent Document's metadata
    # (like 'source' and 'page') onto every resulting child chunk
    chunks = text_splitter.split_documents(documents)

    print(
        f"Split {len(documents)} document(s) into {len(chunks)} chunk(s) "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})."
    )
    return chunks


if __name__ == "__main__":
    # Quick end-to-end test chaining Module 1 -> Module 2
    from document_loader import load_documents

    raw_docs = load_documents()
    chunked_docs = split_documents(raw_docs)

    if chunked_docs:
        print("Chunk 0 metadata:", chunked_docs[0].metadata)
        print("Chunk 0 length:", len(chunked_docs[0].page_content))
        print("Chunk 0 preview:\n", chunked_docs[0].page_content[:250])