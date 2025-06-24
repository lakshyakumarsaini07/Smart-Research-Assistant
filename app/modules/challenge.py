"""
Challenge generation and evaluation utilities
"""
from typing import List, Dict, Any
import json

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from app.utils.config import NUM_CHALLENGES
from app.modules.vector_store import VectorStore
from app.modules.summarizer import LangchainGeminiLLM

def create_challenge_model() -> LangchainGeminiLLM:
    """
    Create an instance of the challenge model
    
    Returns:
        LangchainGeminiLLM model instance
    """
    # Create LangchainGeminiLLM with more creative temperature
    return LangchainGeminiLLM(temperature=0.7)

def generate_challenges(document_chunks: List[Document]) -> List[Dict[str, Any]]:
    """
    Generate challenges based on the document content
    
    Args:
        document_chunks: List of document chunks
        
    Returns:
        List of challenge dictionaries with question and expected answer
    """
    llm = create_challenge_model()
    
    # Combine the document chunks into a single text
    document_text = "\n\n".join([doc.page_content for doc in document_chunks])
    
    # Define a prompt for challenge generation
    challenge_template = f"""
    You are a research comprehension expert tasked with creating challenging questions based on a document.
    
    DOCUMENT TEXT:
    {{text}}
    
    Create {NUM_CHALLENGES} challenging questions based on the document. For each question, provide:
    1. A clear and specific question that tests understanding of the document
    2. The expected answer to the question
    3. A justification explaining where in the document this information can be found
    
    Format your response as a JSON array with objects containing 'question', 'expected_answer', and 'justification' fields.
    Example:
    [
        {{
            "question": "What is the main finding of the study?",
            "expected_answer": "The main finding is that X increases Y by Z%.",
            "justification": "This information is found in the third paragraph of the introduction section."
        }},
        ...
    ]
    """
    
    # Create the prompt
    prompt = PromptTemplate.from_template(challenge_template)
    
    # Use LCEL (LangChain Expression Language) instead of deprecated LLMChain
    chain = prompt | llm
    
    try:
        # Invoke the chain with the document text
        result = chain.invoke({"text": document_text})
        
        # Parse the result
        try:
            challenges = json.loads(result)
            return challenges
        except json.JSONDecodeError:
            print(f"Error parsing challenge response: {result}")
            # Return a default challenge if parsing fails
            return [
                {
                    "question": "What is the main topic of the document?",
                    "expected_answer": "Please read the document and provide your understanding.",
                    "justification": "This is a general question about the document's main topic."
                }
            ]
    except Exception as e:
        print(f"Error generating challenges: {str(e)}")
        # Return a default challenge if there's an error
        return [
            {
                "question": "What is the main topic of the document?",
                "expected_answer": "Please read the document and provide your understanding.",
                "justification": "This is a general question about the document's main topic."
            }
        ]

def evaluate_answer(user_answer: str, challenge: Dict[str, Any], vector_store: VectorStore) -> Dict[str, Any]:
    """
    Evaluate a user's answer to a challenge
    
    Args:
        user_answer: User's answer text
        challenge: Challenge dictionary
        vector_store: Vector store instance
        
    Returns:
        Evaluation dictionary
    """
    # Get the expected answer
    question = challenge["question"]
    expected_answer = challenge["expected_answer"]
    
    # Create an evaluation model
    llm = create_challenge_model()
    
    # Define a prompt for answer evaluation
    evaluation_template = """
    You are an educational assessment expert evaluating a student's answer to a question.
    
    QUESTION:
    {question}
    
    EXPECTED ANSWER:
    {expected_answer}
    
    STUDENT'S ANSWER:
    {user_answer}
    
    Evaluate the student's answer and provide:
    1. Whether the answer is correct ("Yes"), partially correct ("Partially"), or incorrect ("No")
    2. A score from 0 to 100
    3. Helpful feedback for the student
    4. A detailed justification for your evaluation
    
    Format your response as JSON with 'correct', 'score', 'feedback', and 'justification' fields.
    Example:
    {{
        "correct": "Partially",
        "score": 70,
        "feedback": "You correctly identified X, but missed Y.",
        "justification": "The answer includes the main concept but lacks specific details about..."
    }}
    """
    
    # Create the prompt and use LCEL
    prompt = PromptTemplate.from_template(evaluation_template)
    chain = prompt | llm
    
    try:
        # Invoke the chain with the inputs
        result = chain.invoke({
            "question": question,
            "expected_answer": expected_answer,
            "user_answer": user_answer
        })
        
        # Parse the result
        try:
            evaluation = json.loads(result)
            return evaluation
        except json.JSONDecodeError:
            print(f"Error parsing evaluation response: {result}")
            # Return a default evaluation if parsing fails
            return {
                "correct": "Evaluation Failed",
                "score": 0,
                "feedback": "We couldn't automatically evaluate your answer. Please review the expected answer.",
                "justification": "There was a technical issue in processing the evaluation."
            }
    except Exception as e:
        print(f"Error evaluating answer: {str(e)}")
        return {
            "correct": "Evaluation Failed",
            "score": 0,
            "feedback": f"Error evaluating your answer: {str(e)}",
            "justification": "There was a technical issue in processing the evaluation."
        }
