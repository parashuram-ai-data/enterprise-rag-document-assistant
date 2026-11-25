from pathlib import Path
from typing import List
# Standard LangChain schema representing a piece of text + its metadata (source, page)
from langchain_core.documents import Document
# Built-in loaders that handle PDF parsing and plain-text file reading
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(data_dir: str = "data/documents") -> List[Document]:
    """
    Scans the target directory for .pdf and .txt files and loads them
    into a unified list of LangChain Document objects.
    """
    documents: List[Document] = []
    # Pathlib makes cross-platform file handling (Windows '\' vs Linux '/') clean and safe
    data_path = Path(data_dir)

    # Fail fast with a clear error if someone misconfigures the folder path
    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    # Loop through every item inside data/documents
    for file_path in data_path.iterdir():
        # Skip subfolders if any exist; only process actual files
        if not file_path.is_file():
            continue

        # Normalize extension to lowercase so '.PDF' and '.pdf' both work
        ext = file_path.suffix.lower()

        try:
            if ext == ".pdf":
                # PyPDFLoader splits the PDF automatically into 1 Document per page
                # and injects {'source': '<path>', 'page': <int>} into metadata
                loader = PyPDFLoader(str(file_path))
                documents.extend(loader.load())

            elif ext == ".txt":
                # Explicitly set utf-8 encoding to prevent Windows 'cp1252' decode crashes
                loader = TextLoader(str(file_path), encoding="utf-8")
                documents.extend(loader.load())

        except Exception as e:
            # Fault tolerance: if 1 file is corrupted or password-protected,
            # log a warning and keep loading the remaining files instead of crashing
            print(f"[WARN] Failed to load {file_path.name}: {e}")

    print(f"Loaded {len(documents)} page(s)/document(s) from '{data_dir}'.")
    return documents


# This block only runs when you execute `python src/document_loader.py` directly.
# It will NOT run when another file (like rag_pipeline.py) imports `load_documents`.
if __name__ == "__main__":
    docs = load_documents()
    if docs:
        print("Sample metadata:", docs[0].metadata)
        print("Sample preview:", docs[0].page_content[:200])