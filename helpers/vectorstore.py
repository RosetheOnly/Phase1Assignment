from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document

def create_vector_store(docs: List[Document]):
    """
    Creates a Chroma vector store from a list of documents.
    """
    print("Creating vector store...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # ⚡ Fix: Remove persist_directory (use in-memory Chroma)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model
    )

    print("Vector store created successfully.")
    return vectorstore
