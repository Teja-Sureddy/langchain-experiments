"""
Manage vector embeddings of documents or text chunks which allows similarity search or retrieval based on those vectors.

In-memory: InMemoryVectorStore, DocArrayInMemorySearch
In-memory/File-Based/Local: FAISS, Chroma
Database/Local: PgVector, SQLiteVec, Redis, Cassandra
3rd Party: Pinecone, MongoDBAtlasVectorSearch, AzureSearch, DocumentDBVectorSearch
"""
import textwrap
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

# -------------------- In-Memory --------------------
from langchain_community.vectorstores import InMemoryVectorStore, FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def in_memory(vector_type: str = 'faiss'):
    texts = open("../assets/texts.txt", encoding="utf-8").read().splitlines()
    docs = [Document(page_content=t) for t in texts]
    documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store_ins = FAISS if vector_type == 'faiss' else InMemoryVectorStore
    vector_store = vector_store_ins.from_documents(documents=documents, embedding=embeddings)

    while True:
        prompt = input("\nEnter Prompt: ") or "Which animals live in the jungle?"
        relevant_docs = vector_store.similarity_search_with_score(prompt)

        for doc in relevant_docs:
            if (vector_type == 'faiss' and doc[1] < 350) or (vector_type != 'faiss' and doc[1] > 0.6):
                print(textwrap.fill(doc[0].page_content, width=100))


# -------------------- Database --------------------
from langchain_community.vectorstores import SQLiteVec


def database():
    texts = open("../assets/texts.txt", encoding="utf-8").read().splitlines()
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_db = SQLiteVec("my_vectors", None, embeddings, "../tmp/db.sqlite3")
    vector_db.add_texts(texts)

    while True:
        prompt = input("\nEnter Prompt: ") or "Which animals live in the jungle?"
        relevant_docs = vector_db.similarity_search_with_score(prompt)
        for doc in relevant_docs:
            if doc[1] < 20:
                print(textwrap.fill(doc[0].page_content, width=100))


warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

if __name__ == "__main__":
    in_memory()
    # database()
    pass
