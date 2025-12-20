import os
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def create_vector_db():
    pdfs_folder = "."

    documents = []
    for pdf_file in os.listdir(pdfs_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdfs_folder, pdf_file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    texts = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectorstore = FAISS.from_documents(texts, embeddings)

    vectorstore.save_local("faiss_index_nitw")


def retrieve_text(question: str, k: int = 4):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local(
        "faiss_index_nitw", embeddings, allow_dangerous_deserialization=True
    )
    docs = vectorstore.similarity_search(question, k=k)
    return docs


if __name__ == "__main__":
    create_vector_db()