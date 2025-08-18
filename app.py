import streamlit as st
from dotenv import load_dotenv

# Import your helper modules
from helpers.chunker import chunk_data
from helpers.youtubeloader import load_from_youtube
from helpers.vectorstore import create_vector_store
from helpers.retriever import create_retriever
from helpers.chain import create_rag_chain
# If you add other loaders later, import them here (e.g., webloader, pdftloader)

# -------------------------------
# 🌐 Streamlit Config
# -------------------------------
st.set_page_config(page_title="Naive RAG Chatbot", layout="wide")
st.title("🧠 Naive RAG Chatbot")
st.write("Ask questions to your own documents, starting with YouTube transcripts.")

# Load environment variables (API keys, etc.)
load_dotenv()

# -------------------------------
# 🔄 Session State
# -------------------------------
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# -------------------------------
# 📥 Sidebar – Data Source Input
# -------------------------------
with st.sidebar:
    st.header("Setup")
    youtube_url = st.text_input("Enter a YouTube URL:")

    if st.button("Process Source"):
        if youtube_url:
            with st.spinner("Processing video transcript..."):
               
