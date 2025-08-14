# Simple RAG — Sistema de Recuperação de Informação com Manual de Carro

Este projeto implementa um sistema **RAG (Retrieval-Augmented Generation)** que responde a perguntas sobre o manual do veículo **Chevrolet**, utilizando modelos de linguagem e busca vetorial.

---

##  Estrutura do Projeto

| Arquivo / Pasta               | Descrição                                                                 |
|-------------------------------|---------------------------------------------------------------------------|
| `converte_pdf.py`             | Extrai o texto do PDF, divide em páginas e chunks menores, salva em `chunks.json`. |
| `cria_banco.py`               | Carrega os chunks, cria embeddings com `sentence-transformers/all-MiniLM-L6-v2` e armazena no ChromaDB (`chroma_db/`). |
| `chat.py`                     | Interface para conversação em português. Carrega o banco vetorial e o modelo LLM (*llama3 via Ollama*), recupera chunks relevantes e gera respostas com base nesse contexto. |
| `chunks.json`                 | Contém os trechos extraídos e processados do manual.                      |
| `manuais/manual-carro.pdf`    | Manual original do veículo utilizado como base para o sistema.            |

---

##  Fluxo de Funcionamento

1. **Extração do texto** (`converte_pdf.py`): Ler e separar o conteúdo do PDF em chunks utilizáveis.
2. **Criação do banco vetorial** (`cria_banco.py`): Gerar embeddings dos chunks com `all-MiniLM-L6-v2` e armazenar no ChromaDB.
3. **Chat RAG** (`chat.py`): Receber perguntas em português, recuperar os chunks mais relevantes do manual e gerar respostas com base nesse contexto.

---

##  Como Usar

### Pré-requisitos

- Python **3.10+**
- Instale as dependências:
  pip install -r requirements.txt
- Instale o modelo Ollama (llama3) localmente para uso como LLM.

## Passo a passo
Converter o arquivo pdf em chunks
python converte_pdf.py (necessário executar apenas uma vez)

Criar o banco vetorial:
python cria_banco.py (necessário executar apenas uma vez)

Iniciar o chat:
python chat.py (necessário executar toda vez que for utilizar o chat)

## Observações Importantes

As respostas são geradas com base exclusivamente no conteúdo do manual.

O projeto utiliza:

-LangChain

-ChromaDB

-Hugging Face Embeddings

-O sistema opera em português (pt-BR), tanto para entrada quanto para saída.

-O projeto pode ser utilizado com outros arquivos pdf servindo de contexto.

## Exemplos

Pergunta: -  A cada quantos kms deve ser feita a troca de óleo?

Resposta esperada: - A resposta útil é: 7.500 km ou 6 meses, o que ocorrer primeiro, se o veículo estiver sujeito a qualquer destas condições severas de uso. Caso contrário, é recomendado trocar o óleo a cada 15.000 km ou 12 meses, o que ocorrer primeiro.
