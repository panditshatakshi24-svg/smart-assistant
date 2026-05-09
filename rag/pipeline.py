from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from rag.retriever import load_retriever

load_dotenv()

def build_pipeline():
    retriever = load_retriever()
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    return retriever, llm

def ask(query):
    retriever, llm = build_pipeline()

    # Retrieve relevant chunks
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build prompt manually
    prompt = f"""Use the following context to answer the question.
    
Context:
{context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)
    return response.content, docs

if __name__ == "__main__":
    answer, sources = ask("What is UART and why is it used?")
    print("\nAnswer:", answer)
    print("\nSources used:", len(sources))