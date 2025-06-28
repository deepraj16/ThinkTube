import os 
from langchain_openai import ChatOpenAI 
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64 

def setup_llm():
    """Setup and configure the language model"""
    os.environ["OPENAI_API_KEY"] = "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"
    os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
    
    llm = ChatOpenAI(
        model="llama3-70b-8192",   
        temperature=0.7,
    )
    return llm

def create_document_chunks(transcript):
    """Split transcript into smaller chunks"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])  
    return chunks
