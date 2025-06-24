import os 
from langchain_openai import ChatOpenAI 
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
