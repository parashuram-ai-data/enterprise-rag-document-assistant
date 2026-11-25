import time
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from document_loader import load_documents
from text_splitter import split_documents
from vector_store import create_vector_store, load_vector_store

load_dotenv()

RAG_PROMPT_TEMPLATE = """You are an Enterprise Document Assistant. Answer the user's question accurately and concisely using ONLY the retrieved context below.

Rules:
1. Base your answer strictly on the provided context. Do not use outside knowledge or guess.
2. If the context does not contain enough information to answer the question, state clearly: "I cannot find that information in the uploaded documents."
3. Cite the source file name and page number (e.g.,) when referencing facts.

Retrieved Context:
{context}

User Question:
{question}

Answer:"""


def run_ingestion(data_dir: str = "data/documents") -> int:
    docs = load_documents(data_dir)
    if not docs:
        raise ValueError(f"No supported documents (.pdf, .txt) found in '{data_dir}'.")

    chunks = split_documents(docs)
    db = create_vector_store(chunks)
    db.client.close()
    return len(chunks)


def _format_docs_with_citations(docs: List[Document]) -> str:
    formatted_chunks = []
    for doc in docs:
        source_name = Path(doc.metadata.get("source", "Unknown")).name
        page_num = doc.metadata.get("page_label") or (doc.metadata.get("page", 0) + 1)
        header = f""
        formatted_chunks.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_chunks)


def _extract_clean_text(content: Any) -> str:
    """Extracts clean string text when Gemini returns structured content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(text_parts).strip()
    return str(content)


def _invoke_llm_with_fallback(prompt_text: str) -> str:
    models_to_try = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]
    last_error = None

    for model_name in models_to_try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,
        )
        for attempt in range(2):
            try:
                response = llm.invoke(prompt_text)
                return _extract_clean_text(response.content)
            except Exception as e:
                last_error = e
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    time.sleep(2)
                    continue
                break

    raise RuntimeError(f"All Gemini models failed to respond. Last error: {last_error}")


def ask_question(question: str, k: int = 3) -> Dict[str, Any]:
    db = load_vector_store()
    try:
        retrieved_docs = db.similarity_search(question, k=k)

        if not retrieved_docs:
            return {
                "answer": "No relevant context found in the knowledge base.",
                "sources": [],
            }

        context_text = _format_docs_with_citations(retrieved_docs)
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        final_prompt = prompt.format(context=context_text, question=question)

        answer = _invoke_llm_with_fallback(final_prompt)

        return {
            "answer": answer,
            "sources": retrieved_docs,
        }
    finally:
        db.client.close()