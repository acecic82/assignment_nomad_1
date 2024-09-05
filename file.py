import streamlit as st
from pathlib import Path
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

FILE_DIR_PATH = "./.cache/files/"
EMBEDDING_DIR_PATH = "./.cache/embeddings/"

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"{FILE_DIR_PATH}{file.name}"

    try:
        with open(file_path, "wb") as f:
            f.write(file_content)
    except FileNotFoundError:
        Path(FILE_DIR_PATH).mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file_content)

    
    cache_dir = LocalFileStore(f"{EMBEDDING_DIR_PATH}{file.name}")
    
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever