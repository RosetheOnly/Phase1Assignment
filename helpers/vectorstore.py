# helpers/vectorstore.py

from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

def create_vector_store(docs: List[Document]):
    """
    Creates a Chroma vector store from a list of documents.

    Args:
        docs: A list of Document objects (chunks).

    Returns:
        A Chroma vector store instance.
    """
    print("Creating vector store... This may take a moment.")
    
    # Initialize the embedding model from Hugging Face
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # change to 'cuda' if using GPU
    )

    # Create the vector store
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model
    )
    
    print("✅ Vector store created successfully.")
    return vectorstore


def create_retriever(vectorstore, top_k: int = 4) -> VectorStoreRetriever:
    """
    Creates a retriever from a vector store.

    Args:
        vectorstore: A Chroma vector store instance.
        top_k (int): Number of results to retrieve.

    Returns:
        A retriever instance.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    print(f"✅ Retriever created (top_k={top_k}).")
    return retriever
