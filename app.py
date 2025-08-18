import streamlit as st
from dotenv import load_dotenv

# Import your helper functions
from helpers.chunker import chunk_data
from helpers.websiteloader import load_from_website
from helpers.vectorstore import create_vector_store
from helpers.retriever import create_retriever
from helpers.chain import create_rag_chain

# Configure Streamlit
st.set_page_config(page_title="Naive RAG Q&A", layout="wide")
st.title("Naive RAG Chatbot 🤖💬")
st.write("Ask questions based on custom website content!")

# Load environment variables
load_dotenv()

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Sidebar setup
with st.sidebar:
    st.header("Setup")
    website_url = st.text_input("Enter Website URL:")

    if st.button("Process Website"):
        if website_url:
            with st.spinner("Processing website... This may take a few minutes."):
                try:
                    docs = load_from_website(website_url)
                    chunks = chunk_data(docs)
                    vector_store = create_vector_store(chunks)
                    st.session_state.retriever = create_retriever(vector_store)
                    st.session_state.rag_chain = create_rag_chain(st.session_state.retriever)
                    st.success("Website processed successfully!")
                except Exception as e:
                    st.error(f"Error occurred: {e}")
        else:
            st.warning("Please enter a website URL.")

# Main Q&A section
st.header("Q&A")
if st.session_state.rag_chain:
    st.info("Ready to answer questions.")
    question = st.text_input("Ask a question about the website content:")

    if question:
        with st.spinner("Generating answer..."):
            try:
                answer = st.session_state.rag_chain.invoke(question)
                st.write(answer)
            except Exception as e:
                st.error(f"Error generating answer: {e}")
else:
    st.info("Please process a website first using the sidebar.")
