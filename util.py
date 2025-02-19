import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def process_and_store_documents(pdf_paths, save_path):
    """
    Xử lý tài liệu PDF và lưu vào ChromaDB.
    """
    documents = []
    for path in pdf_paths:
        loader = PDFPlumberLoader(path)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=save_path
    )

    return vector_store




def get_retriever(folder_name):
    """
    Trả về retriever từ thư mục tương ứng (Quy trình hoặc Quyết định)
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    try:
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=f"./chroma_db/{folder_name}"
        )

        return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})
    except Exception as e:
        print(f"Lỗi khi khởi tạo vector store: {e}")
        return None

