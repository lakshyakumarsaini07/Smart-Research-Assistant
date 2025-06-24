"""
FastAPI backend for the Smart Research Assistant
"""
import os
import tempfile
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.modules.document_loader import process_document, create_document_chunks
from app.modules.vector_store import VectorStore
from app.modules.summarizer import summarize_document
from app.modules.qa import create_qa_chain, answer_question, ConversationalQA
from app.modules.challenge import generate_challenges, evaluate_answer
from app.utils.config import CHUNK_SIZE, CHUNK_OVERLAP

# Create FastAPI app
app = FastAPI(title="Smart Research Assistant API")

# Add CORS middleware to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store the document and related objects
document_text = None
document_chunks = None
vector_store = VectorStore()
qa_system = None
conversational_qa = None
challenges = None

# Request and response models
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    source: Optional[str] = None

class ChallengeResponse(BaseModel):
    question: str
    id: int

class ChallengeAnswerRequest(BaseModel):
    challenge_id: int
    answer: str

class ChallengeEvaluationResponse(BaseModel):
    correct: str
    score: int
    feedback: str
    justification: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document
    """
    global document_text, document_chunks, vector_store, qa_system, conversational_qa, challenges
    
    # Validate file type
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.pdf', '.txt']:
        raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF and TXT files are supported.")
    
    # Save the uploaded file
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the document
        document_text = process_document(temp_file_path, file_extension)
        
        # Create document chunks
        document_chunks = create_document_chunks(document_text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # Create vector store
        vector_store.create_vector_store(document_chunks)
        
        # Initialize QA system
        qa_system = create_qa_chain(vector_store)
        
        # Initialize conversational QA
        conversational_qa = ConversationalQA(vector_store)
        
        # Generate challenges
        challenges = generate_challenges(document_chunks)
        
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                print(f"Error cleaning up temporary file: {str(cleanup_error)}")
        
        return {"message": "Document uploaded and processed successfully", "filename": file.filename}
    
    except Exception as e:
        # Log detailed error information
        import traceback
        print(f"Error processing document: {str(e)}")
        traceback.print_exc()
        
        # Clean up temporary file if exists
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                print(f"Error cleaning up temporary file: {str(cleanup_error)}")
        
        # Return a more descriptive error
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except OSError as e:
            logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")

@app.get("/summary")
async def get_summary():
    """
    Get a summary of the uploaded document
    """
    global document_chunks
    
    if document_chunks is None:
        raise HTTPException(status_code=400, detail="No document has been uploaded yet")
    
    try:
        summary = summarize_document(document_chunks)
        return {"summary": summary}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Answer a question based on the document
    """
    global qa_system
    
    if qa_system is None:
        raise HTTPException(status_code=400, detail="No document has been uploaded yet")
    
    try:
        result = answer_question(qa_system, request.question)
        
        # Extract source information from the first source document
        source = None
        if result["source_documents"] and len(result["source_documents"]) > 0:
            source_doc = result["source_documents"][0]
            source = source_doc.page_content[:200] + "..." if len(source_doc.page_content) > 200 else source_doc.page_content
        
        return {"answer": result["answer"], "source": source}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@app.post("/ask_with_context", response_model=AnswerResponse)
async def ask_with_context(request: QuestionRequest):
    """
    Answer a question with conversation context
    """
    global conversational_qa
    
    if conversational_qa is None:
        raise HTTPException(status_code=400, detail="No document has been uploaded yet")
    
    try:
        result = conversational_qa.answer_question(request.question)
        
        # Extract source information from the first source document
        source = None
        if result["source_documents"] and len(result["source_documents"]) > 0:
            source_doc = result["source_documents"][0]
            source = source_doc.page_content[:200] + "..." if len(source_doc.page_content) > 200 else source_doc.page_content
        
        return {"answer": result["answer"], "source": source}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@app.get("/challenges", response_model=List[ChallengeResponse])
async def get_challenges():
    """
    Get challenges based on the document
    """
    global challenges
    
    if challenges is None:
        raise HTTPException(status_code=400, detail="No document has been uploaded yet")
    
    try:
        # Format challenges for response
        challenge_responses = [
            {"question": challenge["question"], "id": i} 
            for i, challenge in enumerate(challenges)
        ]
        
        return challenge_responses
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving challenges: {str(e)}")

@app.post("/evaluate_challenge", response_model=ChallengeEvaluationResponse)
async def evaluate_challenge_answer(request: ChallengeAnswerRequest):
    """
    Evaluate a user's answer to a challenge
    """
    global challenges, vector_store
    
    if challenges is None:
        raise HTTPException(status_code=400, detail="No document has been uploaded yet")
    
    if request.challenge_id < 0 or request.challenge_id >= len(challenges):
        raise HTTPException(status_code=400, detail="Invalid challenge ID")
    
    try:
        # Get the challenge
        challenge = challenges[request.challenge_id]
        
        # Evaluate the answer
        evaluation = evaluate_answer(request.answer, challenge, vector_store)
        
        return {
            "correct": evaluation["correct"],
            "score": evaluation["score"],
            "feedback": evaluation["feedback"],
            "justification": evaluation["justification"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating challenge answer: {str(e)}")

# Run with: uvicorn app.core.api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
