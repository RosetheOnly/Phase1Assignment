# helpers/websitesloader.py
"""
Naive RAG Chatbot - Website Loader
----------------------------------
This module loads webpage content from a given URL using LangChain's WebBaseLoader.
It returns a list of LangChain Document objects ready for chunking and vectorization.
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


def load_from_website(url: str) -> list[Document]:
    """
    Loads text content from a website and returns a list of Document objects.

    Args:
        url (str): The URL of the website to load.

    Returns:
        list[Document]: A list of LangChain Document objects containing the page content.
    """
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()

        if not docs:
            raise ValueError(f"No content could be loaded from {url}")

        print(f"✅ Loaded {len(docs)} document(s) from {url}")
        return docs

    except Exception as e:
        print(f"❌ Error loading website: {e}")
        return []
