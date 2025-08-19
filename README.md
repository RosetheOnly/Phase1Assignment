🧠 Naive RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) system that answers user questions based on content scraped from websites.
This project was built as part of the Phase 1 Final Project to demonstrate document ingestion, embedding, retrieval, and generation with LangChain.

🚀 Features

📄 Document Ingestion: Load content from website URLs with WebBaseLoader.

✂️ Chunking: Split large documents into manageable chunks using a recursive character splitter.

🔍 Vector Storage: Embed documents with HuggingFace all-MiniLM-L6-v2 and store vectors in FAISS.

🎯 Retrieval: Search top-k relevant chunks for any user query.

💡 Generation: Provide context-grounded answers using an LLM.

💬 Interactive Q&A: Ask questions in CLI or expand to Streamlit for a web app.

🛠️ Tech Stack

LangChain – framework for document loaders, retrievers, and chains.

FAISS – vector database for fast similarity search.

HuggingFace Sentence Transformers – embeddings (all-MiniLM-L6-v2).

OpenAI GPT (or any LLM) – used for generation.

📂 Project Structure
phase1assignment/
│── app.py
│── requirements.txt
│── helpers/
    │── __init__.py   (can be empty, just to mark it as a package)
    │── websiteloader.py
    │── chunker.py
    │── vectorstore.py
    │── retriever.py
    │── chain.py


⚙️ Setup Instructions

Clone Repo

git clone https://github.com/<your-username>/naive-rag-chatbot.git
cd naive-rag-chatbot


Install Dependencies

pip install -r requirements.txt


Set Environment Variables
If using OpenAI:

export OPENAI_API_KEY="your_api_key_here"


Run Script

python websiteloader.py

🧑‍💻 Usage

Inside websiteloader.py, edit the list of URLs:

urls = [
    "https://example.com/article1",
    "https://example.com/article2"
]


Then, run the script and interact:

> What is the main theme of article1?
Answer: ...

📊 Example Queries

Q: What challenges are highlighted in the article?
A: The article mentions funding gaps, lack of training, and infrastructure issues.

Q: Who are the key stakeholders?
A: The stakeholders include local governments, NGOs, and private investors.

⚠️ Limitations

Uses naive retrieval — no reranking, hybrid search, or advanced filtering.

Dependent on website text quality (boilerplate content may reduce relevance).

Limited grounding — LLM may still hallucinate if relevant chunks aren’t retrieved.

🔮 Future Improvements


✅ Support multiple document sets (switch by topic).

✅ Deploy on HuggingFace Spaces or Streamlit Cloud.

📜 License

MIT License. Free to use, adapt, and extend.
