import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_embeddings():
    """Generate embeddings for documents in the database directory"""
    # Set up directories
    DATABASE_DIR = Path(__file__).resolve().parent.joinpath("data", "database")
    PRETRAINED_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath("data", "pretrained_vector_store")
    
    # Create directories if they don't exist
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    PRETRAINED_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        return False
    
    try:
        # Load documents from database directory
        print("Loading documents...")
        documents = []
        
        # Load PDF files
        pdf_loader = DirectoryLoader(
            DATABASE_DIR.as_posix(), 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader, 
            show_progress=True
        )
        documents.extend(pdf_loader.load())
        
        if not documents:
            print("No documents found in database directory")
            return False
        
        # Split documents into chunks
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings
        print("Creating embeddings...")
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        
        # Create and persist vector store
        print("Creating vector store...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(PRETRAINED_VECTOR_STORE_DIR)
        )
        
        # Persist the vector store
        vector_store.persist()
        
        print("Embeddings generated and stored successfully!")
        return True
        
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        return False

if __name__ == "__main__":
    generate_embeddings() 