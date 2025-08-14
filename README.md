Simple RAG - Recuperação de Informação com Manual de um carro
Este projeto implementa um sistema de perguntas e respostas (RAG - Retrieval Augmented Generation) sobre o manual do veículo Chevrolet Corsa, utilizando modelos de linguagem e busca vetorial.

Estrutura do Projeto
Fluxo de Funcionamento
Extração do texto do PDF

O arquivo converte_pdf.py lê o manual em PDF, divide o texto em páginas e em chunks menores, e salva em chunks.json.
Criação do banco vetorial

O arquivo cria_banco.py carrega os chunks, gera embeddings com o modelo sentence-transformers/all-MiniLM-L6-v2 e persiste no banco vetorial ChromaDB em chroma_db/.
Chat RAG

O arquivo chat.py carrega o banco vetorial e o modelo LLM (llama3 via Ollama), permitindo perguntas em português sobre o manual. O sistema busca os chunks mais relevantes e gera respostas baseadas apenas no contexto encontrado.
Como Usar
Pré-requisitos

Python 3.10+
Instale dependências: pip install -r requirements.txt
Instale o modelo Ollama (llama3) localmente.



Criar o banco vetorial

Iniciar o chat

Digite sua pergunta sobre o manual do carro.
Para sair, digite sair.
Principais Arquivos
chat.py: Interface de perguntas e respostas.
converte_pdf.py: Extrai e divide o texto do PDF.
cria_banco.py: Gera o banco vetorial dos chunks.
chunks.json: Chunks extraídos do manual.
manuais/manual-carro.pdf: Manual original do veículo.
Observações
As respostas são baseadas exclusivamente no conteúdo do manual.
O projeto utiliza LangChain, ChromaDB e HuggingFace Embeddings.
O contexto é retornado em português (pt-BR).
