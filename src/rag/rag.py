import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# pdf_path = Path(__file__).parent/"abc.pdf"
pdf_path = """C:\\Users\\thesh\\Downloads\\Documents\\The+48+Laws+Of+Power.pdf"""

loader = PyPDFLoader(file_path=pdf_path)

docs = loader.load()

# print("Docs",docs)
print("Docs Pages", len(docs))
print("Docs Page 1", docs[1])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=200,
)

split_docs = text_splitter.split_documents(documents=docs)

print("Split Docs", len(split_docs))

# set OPENAI_API_KEY as env variable
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large", api_key=OPENAI_API_KEY
)

print("Embedding Docs", embeddings.model)

# Adding chunk to DB
# vector_store =  QdrantVectorStore.from_documents(
#     documents=[],
#     collection_name="learn_langchain",
#     url="http://localhost:6333",
#     embedding=embeddings
# )
#
#
# vector_store.add_documents(documents=split_docs)
print("Injection done, Vector DB")

# Retrieval
retriever = QdrantVectorStore.from_existing_collection(
    collection_name="learn_langchain",
    url="http://localhost:6333",
    embedding=embeddings
)

userQuery = "What we learn about TRUST IN FRIENDS ?"
search_result = retriever.similarity_search(
    query=userQuery
)

print("Relevant Chunks Search Result", search_result)

system_prompt =f"""
You are an helpful AI Assistant who responds base of the available context.
Context: {search_result}

"""

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":userQuery}
    ],
)

print("Response=>\n", response.choices[0].message.content, "\n\n")




