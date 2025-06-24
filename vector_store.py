import os
from langchain_community.vectorstores import Chroma
# from langchain_community.vectorstores import DocArrayInMemorySearch
# from langchain_community.vectorstores import Qdrant
from langchain_together import TogetherEmbeddings

def setup_embeddings_and_vectorstore(chunks):
    """Setup embeddings and create Chroma vector store"""
    os.environ["OPENAI_API_KEY"] = "2f2c9e25075a8768b707adcff56865283eb487cb962e920fbcc6f432cbb001f7"
    os.environ["OPENAI_API_BASE"] = "https://api.together.xyz/v1"
    
    embeddings = TogetherEmbeddings(
        model="BAAI/bge-base-en-v1.5"
    )
 
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    
    return vector_store
