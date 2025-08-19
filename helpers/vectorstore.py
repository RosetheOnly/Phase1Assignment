# helpers/vectorstore.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document

def create_vector_store(docs: List[Document]):
    """
    Creates a Chroma vector store from a list of documents.
    (Uses in-memory storage to avoid SQLite version issues.)

    Args:
        docs: A list of Document objects (chunks).

    Returns:
        A Chroma vector store instance.
    """
    print("Creating vector store... This may take a moment.")

    # Initialize the embedding model from Hugging Face
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # set to 'cuda' if GPU available
    )

    # ✅ No persist_directory → avoids SQLite compatibility issues
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model
    )

    print("Vector store created successfully.")
    return vectorstore


def create_retriever(vectorstore):
    """
    Creates a retriever from the Chroma vector store.

    Args:
        vectorstore: The Chroma vector store instance.

    Returns:
        A retriever object for similarity search.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # number of relevant chunks to retrieve
    )
    return retriever
