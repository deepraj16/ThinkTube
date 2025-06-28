import os
from langchain_community.vectorstores import Chroma
# from langchain_community.vectorstores import DocArrayInMemorySearch
# from langchain_community.vectorstores import Qdrant
from langchain_together import TogetherEmbeddings
import numpy as np
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64 


def setup_embeddings_and_vectorstore(chunks):
    """Setup embeddings and create Chroma vector store"""
    os.environ["OPENAI_API_KEY"] = "268ac4c5c255e13be7ed21abe9f14e595a748ce1dc5139a10b5523fd2db42f8b"
    os.environ["OPENAI_API_BASE"] = "https://api.together.xyz/v1"
    
    embeddings = TogetherEmbeddings(
        model="BAAI/bge-base-en-v1.5"
    )
 
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    
    return vector_store
