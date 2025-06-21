"""
RAG - Retrieval-Augmented Generation
"""

import textwrap
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import SQLiteVec
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


def main():
    # Document Loader, Text Splitter
    loader = PyMuPDFLoader("../assets/django.pdf")
    pages = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(pages)

    # LLM, Embedding
    llm = Ollama(model="gemma3")
    embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    # Vector Store, Retriever
    vector_store = SQLiteVec("django", None, embedding, "../tmp/db.sqlite3")
    vector_store.add_documents(documents)
    retriever = vector_store.as_retriever(kwargs={'k': 10, 'score_threshold': 0.7})

    # Chain
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    while True:
        prompt = input("\nEnter Prompt: ") or "What are signals and give me an example" or "list all types of password hashers"
        response = chain.run(prompt)
        print(textwrap.fill(response, width=100))


if __name__ == "__main__":
    main()
