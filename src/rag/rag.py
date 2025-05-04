import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import  QdrantVectorStore

load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_KEY")

# pdf_path = Path(__file__).parent/"abc.pdf"
pdf_path = "/Users/shivam/Downloads/The-48-Laws-of-Power-Robert-Greene.pdf"

loader = PyPDFLoader(file_path=pdf_path)

docs = loader.load()

# print("Docs",docs)
print("Docs Pages",len(docs))
print("Docs Page 1",docs[1])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=200,
)

split_docs = text_splitter.split_documents(documents=docs)

print("Split Docs",len(split_docs))

# set OPENAI_API_KEY as env variable
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-large",
)

print("Embedding Docs",embeddings.model)

QdrantVectorStore.add(collection_name="learn_langchain",)

vector_store =  QdrantVectorStore.from_existing_collection(
    collection_name="learn_langchain",
    url="http://localhost:6333",
    embedding=embeddings
)

vector_store.add_documents(documents=split_docs)
print("Injection done, Vector DB")







