# websiteloader.py
"""
Naive RAG Chatbot over Website URLs
-----------------------------------
Loads webpages, chunks them, builds a FAISS vector index,
and allows interactive Q&A using LangChain + HuggingFace embeddings.
"""

import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI  # You can swap with ChatOpenAI or another LLM

# ----------------------------
# CONFIG
# ----------------------------
DB_DIR = "faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # light & fast
GEN_MODEL = "gpt-3.5-turbo"  # replace with your chosen LLM

# ----------------------------
# HELPERS
# ----------------------------

def load_web
