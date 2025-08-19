# helpers/vectorstore.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document


def create_vector_store(docs: List[Document], persist_directory: str = None):
    """
    Creates a Chroma vector store from a list of documents.
    Optionally persists to disk.
    """
    print("Creating vector store... This may take a moment.")

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Change to 'cuda' if GPU available
    )

    if persist_directory:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        vectorstore.persist()
    else:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model
        )

    print("Vector store created successfully.")
    return vectorstore


def create_retriever(vectorstore, search_kwargs={"k": 4}):
    """
    Creates a retriever from a given vectorstore.
    """
    return vectorstore.as_retriever(search_kwargs=search_kwargs)
