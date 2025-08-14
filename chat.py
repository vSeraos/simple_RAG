import torch
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# --- Configurações ---
CHROMA_DIR = "chroma_db"
LLM_MODEL = "llama3"

def run_rag_query_loop(chroma_dir, llm_model):

    device = "cpu"

    # Carregar Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )

    # Carregar banco vetorial
    db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)

    # Carregar LLM
    llm = Ollama(model=llm_model)

    # Configurar Retriever
    # K=5 é um bom valor, mas pode ajustar se precisar de mais contexto.
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # Instruções do contexto.
    template = """
    Use APENAS o contexto fornecido abaixo para responder à pergunta.
    Se a resposta não puder ser encontrada no contexto, responda "Não consegui encontrar a resposta no contexto fornecido."
    O documento se trata de uma manual de veículo modelo Corsa ano 2008.
    Responda em pt-BR.

    Contexto:
    {context}

    Pergunta: {question}

    Resposta útil:
    """

    prompt = PromptTemplate.from_template(template)

    # A cadeia de QA usa o template de prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    
    while True:
        pergunta = input("\n Pergunta: ")
        if pergunta.lower() == "sair":
            break

        print(" Pensando...")
        resposta = qa_chain.invoke({"query": pergunta})
        
        print("\n Minha Resposta é:", resposta["result"])
        
    
if __name__ == "__main__":
    run_rag_query_loop(CHROMA_DIR, LLM_MODEL)