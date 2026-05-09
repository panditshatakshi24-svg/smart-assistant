from langchain_community.embeddings import FakeEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
embeddings = FakeEmbeddings(size=384)
embeddings = FakeEmbeddings(size=384)
def load_retriever(index_path="rag/index", k=3):

    # Load the same embedding model used during ingestion
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load the saved FAISS index from disk
    vectorstore = FAISS.load_local(
        index_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    # k = how many chunks to retrieve per query
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

if __name__ == "__main__":
    retriever = load_retriever()
    query = "What is UART?"
    results = retriever.invoke(query)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content)