from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_vector_store():
    embeddings = HuggingFaceEmbeddings()
    return FAISS.load_local("embeddings", embeddings)

def query_rag(vector_store, query):
    docs = vector_store.similarity_search(query, k=3)
    context = " ".join([d.page_content for d in docs])
    return context
