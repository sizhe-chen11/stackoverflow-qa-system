from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, 
                 chunk_size=500, 
                 chunk_overlap=50,
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Use local embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )
        
    def split_documents(self, documents):
        """Split documents into smaller chunks"""
        logger.info("Splitting documents...")
        texts = self.text_splitter.split_documents(documents)
        logger.info(f"Document splitting completed, total {len(texts)} chunks")
        return texts
    
    def create_vectorstore(self, texts, persist_directory="./vectorstore"):
        """Create vector database"""
        logger.info("Creating vector database...")
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        logger.info("Vector database creation completed")
        return vectorstore
    
    def load_vectorstore(self, persist_directory="./vectorstore"):
        """Load existing vector database"""
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            logger.info("Loading existing vector database...")
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            return vectorstore
        else:
            logger.warning("Vector database does not exist")
            return None