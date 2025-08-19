# helpers/chunker.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def chunk_data(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits documents into smaller chunks for embedding & retrieval.

    Args:
        docs (List[Document]): List of input documents.
        chunk_size (int): Max size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[Document]: List of chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(docs)
    print(f"✅ Split {len(docs)} docs into {len(chunks)} chunks.")
    return chunks
