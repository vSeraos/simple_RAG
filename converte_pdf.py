import os
import json
import fitz
from multiprocessing import Pool
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_FOLDER = "manuais"
OUTPUT_FILE = "chunks.json"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

def parse_pdf(file_path: str) -> list[Document]:
    try:
        doc = fitz.open(file_path)
        pages_as_docs = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                metadata = {"source": file_path, "page": page_num + 1}
                pages_as_docs.append(Document(page_content=text, metadata=metadata))
        return pages_as_docs
    except Exception:
        return []

def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return text_splitter.split_documents(documents)

if __name__ == "__main__":
    pdf_files = [os.path.join(PDF_FOLDER, f) for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        exit()

    with Pool() as pool:
        results = list(pool.imap(parse_pdf, pdf_files))
    all_docs = [doc for doc_list in results for doc in doc_list]

    with Pool() as pool:
        chunked_docs_list = list(pool.imap(split_documents, [[doc] for doc in all_docs]))
    all_chunks = [chunk for sublist in chunked_docs_list for chunk in sublist]

    output_data = [{"page_content": d.page_content, "metadata": d.metadata} for d in all_chunks]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)