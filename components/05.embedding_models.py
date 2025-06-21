"""
Converts input text into a fixed-size dense numeric vector or embedding.

Free: SentenceTransformerEmbeddings, OllamaEmbeddings, HuggingFaceEmbeddings
Paid: OpenAIEmbeddings, AzureOpenAIEmbeddings, CohereEmbeddings, VoyageEmbeddings
"""
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_community.embeddings import SentenceTransformerEmbeddings, OllamaEmbeddings


def get_vectors(embedding: str = 'ollama'):
    texts = open("../assets/texts.txt", encoding="utf-8").read().splitlines()[:1]
    embedder = OllamaEmbeddings(model='nomic-embed-text') if embedding == 'ollama' else SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    vectors = embedder.embed_documents(texts)
    print(f"Embedding: {vectors[0][:5]}\nDimensions: {len(vectors[0])}")


warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

if __name__ == "__main__":
    get_vectors()
