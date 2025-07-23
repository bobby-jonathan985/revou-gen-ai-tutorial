from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

def chunk_document(file_path, chunk_size=500, chunk_overlap=100):
  loader = PyMuPDFLoader(file_path)
  pages = loader.load()

  splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=['\n\n', '\n', ' ', '.', ''],
  )

  chunks = splitter.split_documents(pages)
  return chunks

def save_to_vector(chunks, dir_path, store_name):
  embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
  Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=dir_path,
    collection_name=store_name,
  )

def load_vector_db(dir_path, store_name):
  embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
  vector_db = Chroma(
    persist_directory=dir_path,
    collection_name=store_name,
    embedding_function=embeddings,
  )

  return vector_db

def build_models():
  chunks = chunk_document('data/source.pdf')
  save_to_vector(chunks, 'data', 'mini_project_vector')