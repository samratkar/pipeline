"""
Ingest documents into the vector store.

Run once (or re-run after adding new documents):
    python ingest.py
"""

import os
import sys
import shutil

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import config


def load_documents(documents_dir: str):
    """Load all PDFs from the documents directory."""
    pdf_files = [
        f for f in os.listdir(documents_dir) if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        print(f"No PDF files found in '{documents_dir}'.")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF file(s):")
    all_docs = []
    for filename in sorted(pdf_files):
        path = os.path.join(documents_dir, filename)
        print(f"  Loading: {filename}")
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        # Tag each page with the source filename for later citation
        for doc in docs:
            doc.metadata["source_file"] = filename
        all_docs.extend(docs)
        print(f"    → {len(docs)} page(s)")

    print(f"\nTotal pages loaded: {len(all_docs)}")
    return all_docs


def split_documents(docs):
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks after splitting: {len(chunks)}")
    return chunks


def build_vector_store(chunks):
    """Embed chunks and persist to ChromaDB."""
    embeddings = OpenAIEmbeddings(
        model=config.OPENAI_EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )

    # Remove existing DB so we start fresh on re-ingest
    if os.path.exists(config.CHROMA_DB_DIR):
        print(f"\nRemoving existing vector store at '{config.CHROMA_DB_DIR}'...")
        shutil.rmtree(config.CHROMA_DB_DIR)

    print("Embedding chunks and building vector store (this may take a minute)...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DB_DIR,
    )
    print(f"Vector store saved to '{config.CHROMA_DB_DIR}'.")
    return vector_store


def main():
    print("=== RAG Ingestion Pipeline ===\n")
    docs = load_documents(config.DOCUMENTS_DIR)
    chunks = split_documents(docs)
    build_vector_store(chunks)
    print("\nIngestion complete. Run 'python query.py' or 'streamlit run app.py' to query.")


if __name__ == "__main__":
    main()
