# helpers/vectorstore.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document


def create_vector_store(docs: List[Document], persist_directory: str = "chroma_store"):
    """
    Creates (and persists) a Chroma vector store from a list of documents.

    Args:
        docs (List[Document]): A list of Document objects (chunks).
        persist_directory (str): Directory to store the Chroma DB locally.

    Returns:
        Chroma: A Chroma vector store instance.
    """
    print("⚡ Creating vector store... This may take a moment.")

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # full HuggingFace model name
        model_kwargs={'device': 'cpu'}  # set 'cuda' if you have a GPU
    )

    # Create or load Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    # Persist the DB to disk (optional but useful for reloading later)
    vectorstore.persist()

    print(f"✅ Vector store created and persisted at '{persist_directory}'.")
    return vectorstore
