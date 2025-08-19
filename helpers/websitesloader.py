# app.py
import streamlit as st
from dotenv import load_dotenv

# Import your helpers
from helpers.websitesloader import load_from_website
from helpers.chunker import chunk_data
from helpers.vectorstore import create_vector_store, create_retriever
from helpers.chain import create_rag_chain

# Load environment variables (for API keys, e.g., OpenAI)
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Naive RAG Chatbot", layout="wide")
st.title("Naive RAG Chatbot 🌐")
st.write("Ask questions over custom **websites** using Retrieval-Augmented Generation (RAG).")

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- Sidebar Input ---
with st.sidebar:
    st.header("Setup")
    website_url = st.text_input("Enter Website URL:")

    if st.button("Process Website"):
        if website_url:
            with st.spinner("Loading and processing website..."):
                try:
                    # Step 1: Load website
                    docs = load_from_website(website_url)

                    # Step 2: Chunk data
                    chunks = chunk_data(docs)

                    # Step 3: Create vector store
                    vector_store = create_vector_store(chunks)

                    # Step 4: Create retriever
                    st.session_state.retriever = create_retriever(vector_store)

                    # Step 5: Build RAG chain
                    st.session_state.rag_chain = create_rag_chain(st.session_state.retriever)

                    st.success("✅ Website processed successfully! You can now ask questions.")

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a website URL.")

# --- Main Q&A Area ---
st.header("Ask Questions")
if st.session_state.rag_chain:
    question = st.text_input("Type your question here:")
    if question:
        with st.spinner("Generating answer..."):
            try:
                answer = st.session_state.rag_chain.invoke(question)
                st.subheader("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("➡️ Enter a website URL in the sidebar and click 'Process Website' to get started.")
