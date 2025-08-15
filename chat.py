from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# --- Configurações ---
CHROMA_DIR = "chroma_db"
LLM_MODEL = "llama3"

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Banco vetorial
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# LLM local via Ollama
llm = Ollama(model=LLM_MODEL)

# Prompt
template = """
Use APENAS o contexto fornecido para responder à pergunta.
Se não encontrar a resposta, diga: "Não consegui encontrar a resposta no contexto fornecido."
Documento: Manual do Corsa 2008.
Responda em pt-BR.

Contexto:
{context}

Pergunta: {question}

Resposta útil:
"""
prompt = PromptTemplate.from_template(template)

# Cadeia de QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)

# Loop de perguntas
while True:
    pergunta = input("\nPergunta: ")
    if pergunta.lower() == "sair":
        break
    resposta = qa_chain.invoke({"query": pergunta})
    print("\nMinha resposta é:", resposta["result"])
