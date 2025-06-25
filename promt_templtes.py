
from langchain.prompts import PromptTemplate
import numpy as np
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64 

def create_prompt_template():
    """Create the prompt template for the RAG system"""
    prompt = PromptTemplate(
        template=""" 
            You are a helpful assistant. 
            Answer ONLY from the provided transcript context. 
            If the context is insufficient, just say you don't know. 
            {context} 
            Question: {question}
        """, 
        input_variables=['context', 'question']
    )
    return prompt
