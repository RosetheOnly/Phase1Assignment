# helpers/chain.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Try Groq first, fallback to OpenAI
try:
    from langchain_groq import ChatGroq
    USE_GROQ = True
except ImportError:
    from langchain_openai import ChatOpenAI
    USE_GROQ = False


def _format_docs(docs: list) -> str:
    """
    Formats the retrieved documents into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever):
    """
    Creates the full RAG chain for question answering.
    Falls back to OpenAI if Groq is unavailable.
    """
    if USE_GROQ:
        llm = ChatGroq(
            temperature=0,
            model_name="llama3-8b-8192"
        )
    else:
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo"
        )

    # Prompt
    prompt_template = """
    Answer the following question based only on the provided context.
    If the context does not contain the answer, say:
    "The answer is not available in the provided context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Build the RAG chain
    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
