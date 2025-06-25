"""
It fetches relevant documents/data in response to a user query/prompt.

These documents can come from,
Own sources: (you ingest the data)
 - SVMRetriever - in-memory vector-based search
 - For other vector stores , use .as_retriever()
 - BM25Retriever - in-memory keyword/text-based search
 - TFIDFRetriever - in-memory frequency-based search

External sources: (data may already exist, or may require ingestion)
 - AskNewsRetriever - Latest news from news API
 - TavilySearchAPIRetriever - Internet search results, returns multiple results with score and source
 - WikipediaRetriever - Wikipedia articles
 - AmazonKendra, AzureAISearch, GoogleVertexAISearch Retrievers - Requires ingesting data (storage, db, files, ...)
"""
import textwrap
from dotenv import load_dotenv
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

# -------------------- External --------------------
from langchain_ollama import ChatOllama
from langchain_community.retrievers import TavilySearchAPIRetriever, WikipediaRetriever
from langchain.chains.summarize import load_summarize_chain


def external(model: str = 'gemma3', source: str = 'wiki'):
    llm = ChatOllama(model=model)
    retriever = TavilySearchAPIRetriever() if source == 'tavily' else WikipediaRetriever()

    while True:
        prompt = input("\nEnter Prompt: ") or "What is python?"
        docs = retriever.invoke(prompt)
        summary = load_summarize_chain(llm).run(docs)
        print(textwrap.fill(summary, width=100))


# -------------------- Own --------------------
from langchain_community.retrievers import SVMRetriever
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document


def own():
    texts = open("../assets/texts.txt", encoding="utf-8").read().splitlines()
    documents = [Document(page_content=t) for t in texts]
    retriever = SVMRetriever.from_documents(documents, OllamaEmbeddings(model="nomic-embed-text"))

    while True:
        prompt = input("\nEnter Prompt: ") or "vegetables"
        results = retriever.get_relevant_documents(prompt)
        for doc in results:
            print(textwrap.fill(doc.page_content, width=100))


load_dotenv()
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

if __name__ == "__main__":
    external()
    # own()
