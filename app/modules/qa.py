"""
Question answering utilities
"""
from typing import List, Dict, Any, Optional

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory

from app.utils.llm_interface import GeminiLLM
from app.modules.vector_store import VectorStore
from app.modules.summarizer import LangchainGeminiLLM

def create_qa_model() -> LangchainGeminiLLM:
    """
    Create an instance of the QA model
    
    Returns:
        LangchainGeminiLLM model instance
    """
    # Create GeminiLLM with higher temperature for more diverse answers
    return LangchainGeminiLLM(temperature=0.7)

def create_qa_chain(vector_store: VectorStore) -> RetrievalQA:
    """
    Create a QA chain for document-based question answering
    
    Args:
        vector_store: Vector store instance
        
    Returns:
        RetrievalQA chain
    """
    llm = create_qa_model()
    
    # Create a retriever from the vector store
    retriever = vector_store.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Define a custom prompt for QA
    template = """
    You are a research assistant providing information based solely on the given document.
    
    IMPORTANT: Base your answers strictly on the provided document context.
    If the information cannot be found in the context, say "I don't have enough information to answer this question based on the document."
    Do not make up or infer information not present in the document.
    
    ALWAYS include a "Justification" section that explains where in the document you found the information (e.g., "Based on paragraph X from section Y...").
    
    Context from the document:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    QA_PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
        verbose=False
    )
    
    return qa_chain

def answer_question(qa_chain: RetrievalQA, question: str) -> Dict[str, Any]:
    """
    Answer a question based on the document
    
    Args:
        qa_chain: RetrievalQA chain
        question: Question to answer
        
    Returns:
        Dict containing the answer and source documents
    """
    result = qa_chain({"query": question})
    
    # Extract the answer and source documents
    answer = result.get("result", "I couldn't find an answer to your question.")
    source_docs = result.get("source_documents", [])
    
    return {
        "answer": answer,
        "source_documents": source_docs
    }

class ConversationalQA:
    """
    Class for conversational QA with memory
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize the conversational QA system
        
        Args:
            vector_store: Vector store instance
        """
        self.llm = create_qa_model()
        self.vector_store = vector_store
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.conversation_history = []
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question in a conversational context
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary with answer and source
        """
        # Get relevant documents for the question
        docs = self.vector_store.vector_store.similarity_search(question, k=4)
        
        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Add current question to history
        self.conversation_history.append({"role": "user", "content": question})
        
        # Build conversation history string
        history_str = ""
        for exchange in self.conversation_history:
            history_str += f"{exchange['role'].upper()}: {exchange['content']}\n"
        
        # Create prompt with context and history
        prompt = f"""
        You are a helpful research assistant answering questions based on the provided documents.
        
        CONVERSATION HISTORY:
        {history_str}
        
        CONTEXT FROM DOCUMENTS:
        {context}
        
        Answer the most recent question based on the conversation history and context.
        If you don't know the answer from the provided information, say "I don't have enough information to answer this question."
        
        ANSWER:
        """
        
        # Generate answer
        llm_instance = GeminiLLM(temperature=0.7)
        answer = llm_instance.generate(prompt)
        
        # Add answer to history
        self.conversation_history.append({"role": "assistant", "content": answer if answer else "I couldn't generate a response."})
        
        return {
            "answer": answer if answer else "I couldn't generate a response.",
            "source_documents": docs
        }
