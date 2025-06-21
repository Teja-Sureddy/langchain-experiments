"""
Ingest data from various sources into LangChain Documents.

Sources can be,
Webpage - WebBaseLoader
File - TextLoader, CSV, JSON, PyMuPDF, Blob, DataFrame
Unstructured File - UnstructuredCSVLoader, PDF, Excel, Email, File, HTML, Image, URL, WordDocument, XML
Database - SQLDatabaseLoader, Mongodb, Cassandra
Cloud - S3FileLoader, AzureBlobStorageFileLoader, GCSFileLoader, GoogleDriveLoader
Productivity - FigmaFileLoader, GithubFileLoader
Others - RedditPostsLoader, WhatsAppChatLoader
"""
import os
import textwrap
import warnings
from dotenv import load_dotenv
from langchain_core._api.deprecation import LangChainDeprecationWarning

# -------------------- File --------------------
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain


def file(loader_type: str = 'pdf', model: str = 'gemma3'):
    loader = PyMuPDFLoader('../assets/invoice.pdf') if loader_type == 'pdf' else CSVLoader('../assets/grades.csv')
    docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())

    llm = ChatOllama(model=model)
    chain = load_qa_chain(llm, chain_type="stuff")

    while True:
        prompt = input('\nEnter prompt: ') or ("whats the total cost?" if loader_type == 'pdf'
                                               else 'grade A+ > A > A- > B+ ..., who has the highest grade?')
        response = chain.run(input_documents=docs, question=prompt)
        print(textwrap.fill(response, width=100))


# -------------------- Database --------------------
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.document_loaders import SQLDatabaseLoader


def database(model: str = 'gemma3'):
    user, password = os.environ.get('POSTGRES_USERNAME'), os.environ.get('POSTGRES_PASSWORD')
    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@localhost:5432/langchain")
    loader = SQLDatabaseLoader(query="select * from users", db=SQLDatabase(engine))
    docs = loader.load()

    llm = ChatOllama(model=model)
    chain = load_qa_chain(llm, chain_type="stuff")

    while True:
        prompt = input('\nEnter prompt: ') or "List all names whose age is >= 40"
        response = chain.run(input_documents=docs, question=prompt)
        print(textwrap.fill(response, width=100))


load_dotenv()
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

if __name__ == "__main__":
    file()
    # file('csv')
    # database()
    pass
