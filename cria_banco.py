import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

CHUNKS_FILE = "chunks.json"
CHROMA_DIR = "chroma_db"

# 1. Carregar chunks
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

docs = [Document(page_content=chunk["page_content"], metadata=chunk["metadata"]) for chunk in chunks]

# 2. Criar banco vetorial
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'} 
)
print(f" Gerando embeddings para {len(docs)} chunks...")

texts = [doc.page_content for doc in docs]
metadatas = [doc.metadata for doc in docs]

vectorstore = Chroma.from_texts(
    texts,
    embedding=embeddings_model,
    metadatas=metadatas,
    persist_directory=CHROMA_DIR
)

vectorstore.persist()

print("Banco criado com sucesso!")