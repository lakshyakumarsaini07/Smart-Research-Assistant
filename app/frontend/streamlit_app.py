"""
Streamlit frontend for the Smart Research Assistant
"""
import streamlit as st
import requests
import json
import os
from typing import List, Dict, Any

# API endpoint URL
API_URL = "http://localhost:8000"

# Set page config
st.set_page_config(
    page_title="Smart Research Assistant",
    page_icon="🧠",
    layout="wide"
)

# Helper functions
def upload_document(file):
    """
    Upload a document to the API
    """
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_URL}/upload", files=files)
        
        # Check response status
        if response.status_code != 200:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                if "detail" in error_data:
                    error_detail = error_data["detail"]
            except json.JSONDecodeError:
                error_detail = response.text if response.text else f"HTTP error {response.status_code}"
            
            raise Exception(f"API Error: {error_detail}")
        
        # Try to parse JSON response
        try:
            return response.json()
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from server")
    except requests.exceptions.ConnectionError:
        raise Exception("Failed to connect to API server. Make sure the server is running.")
    except Exception as e:
        raise Exception(f"Error: {str(e)}")

def get_summary():
    """
    Get the document summary from the API
    """
    try:
        response = requests.get(f"{API_URL}/summary")
        
        # Check response status
        if response.status_code != 200:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                if "detail" in error_data:
                    error_detail = error_data["detail"]
            except json.JSONDecodeError:
                error_detail = response.text if response.text else f"HTTP error {response.status_code}"
            
            raise Exception(f"API Error: {error_detail}")
        
        # Try to parse JSON response
        try:
            return response.json()
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from server")
    except requests.exceptions.ConnectionError:
        raise Exception("Failed to connect to API server. Make sure the server is running.")
    except Exception as e:
        raise Exception(f"Error: {str(e)}")

def ask_question(question, with_context=False):
    """
    Ask a question to the API
    """
    try:
        endpoint = "/ask_with_context" if with_context else "/ask"
        response = requests.post(
            f"{API_URL}{endpoint}", 
            json={"question": question}
        )
        
        # Check response status
        if response.status_code != 200:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                if "detail" in error_data:
                    error_detail = error_data["detail"]
            except json.JSONDecodeError:
                error_detail = response.text if response.text else f"HTTP error {response.status_code}"
            
            raise Exception(f"API Error: {error_detail}")
        
        # Try to parse JSON response
        try:
            return response.json()
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from server")
    except requests.exceptions.ConnectionError:
        raise Exception("Failed to connect to API server. Make sure the server is running.")
    except Exception as e:
        raise Exception(f"Error: {str(e)}")

def get_challenges():
    """
    Get challenges from the API
    """
    try:
        response = requests.get(f"{API_URL}/challenges")
        
        # Check response status
        if response.status_code != 200:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                if "detail" in error_data:
                    error_detail = error_data["detail"]
            except json.JSONDecodeError:
                error_detail = response.text if response.text else f"HTTP error {response.status_code}"
            
            raise Exception(f"API Error: {error_detail}")
        
        # Try to parse JSON response
        try:
            return response.json()
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from server")
    except requests.exceptions.ConnectionError:
        raise Exception("Failed to connect to API server. Make sure the server is running.")
    except Exception as e:
        raise Exception(f"Error: {str(e)}")

def evaluate_challenge_answer(challenge_id, answer):
    """
    Evaluate a challenge answer
    """
    try:
        response = requests.post(
            f"{API_URL}/evaluate_challenge",
            json={"challenge_id": challenge_id, "answer": answer}
        )
        
        # Check response status
        if response.status_code != 200:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                if "detail" in error_data:
                    error_detail = error_data["detail"]
            except json.JSONDecodeError:
                error_detail = response.text if response.text else f"HTTP error {response.status_code}"
            
            raise Exception(f"API Error: {error_detail}")
        
        # Try to parse JSON response
        try:
            return response.json()
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from server")
    except requests.exceptions.ConnectionError:
        raise Exception("Failed to connect to API server. Make sure the server is running.")
    except Exception as e:
        raise Exception(f"Error: {str(e)}")

# Initialize session state variables
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "summary" not in st.session_state:
    st.session_state.summary = None
if "mode" not in st.session_state:
    st.session_state.mode = None
if "challenges" not in st.session_state:
    st.session_state.challenges = None
if "challenge_answers" not in st.session_state:
    st.session_state.challenge_answers = {}
if "challenge_evaluations" not in st.session_state:
    st.session_state.challenge_evaluations = {}
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# Page header
st.title("🧠 Smart Research Assistant")
st.markdown("""
Upload a document (PDF or TXT) to get a summary, ask questions, or test your comprehension.
This assistant helps you understand documents better through AI-powered analysis.
""")

# Document upload section
with st.expander("📄 Document Upload", expanded=not st.session_state.document_uploaded):
    uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])
    
    if uploaded_file is not None and st.button("Process Document"):
        with st.spinner("Processing document..."):
            try:
                result = upload_document(uploaded_file)
                st.session_state.document_uploaded = True
                st.success(f"Document '{uploaded_file.name}' processed successfully!")
                
                # Get summary
                with st.spinner("Generating summary..."):
                    summary_result = get_summary()
                    st.session_state.summary = summary_result["summary"]
                
                # Get challenges
                with st.spinner("Generating challenges..."):
                    st.session_state.challenges = get_challenges()
                
                # Reset other session variables
                st.session_state.mode = None
                st.session_state.challenge_answers = {}
                st.session_state.challenge_evaluations = {}
                st.session_state.qa_history = []
                
                # Rerun to update the UI
                st.session_state["rerun"] = True
            
            except Exception as e:
                error_msg = str(e)
                
                # Provide more helpful error messages for common issues
                if "No text could be extracted" in error_msg:
                    if "scanned document" in error_msg or "image-based" in error_msg:
                        st.error(
                            "Error: This appears to be a scanned or image-based PDF.\n\n"
                            "Please ensure you have installed OCR dependencies:\n"
                            "1. Install Python packages: `pip install pytesseract pdf2image Pillow`\n"
                            "2. Install Tesseract OCR: [Download here](https://github.com/UB-Mannheim/tesseract/wiki) (Windows) or via package manager (Linux/Mac)\n\n"
                            "Full error: " + error_msg
                        )
                    elif "encrypted" in error_msg or "password-protected" in error_msg:
                        st.error(
                            " Error: This PDF appears to be encrypted or password-protected.\n\n"
                            "Please try removing the password protection before uploading."
                        )
                    else:
                        st.error(f" Error processing document: {error_msg}")
                elif "Tesseract OCR is not installed" in error_msg:
                    st.error(
                        " Error: Tesseract OCR is required but not installed.\n\n"
                        "Please install Tesseract OCR:\n"
                        "- Windows: [Download here](https://github.com/UB-Mannheim/tesseract/wiki)\n"
                        "- Linux: `sudo apt-get install tesseract-ocr`\n"
                        "- Mac: `brew install tesseract`"
                    )
                elif "Missing Python dependencies" in error_msg and ("pytesseract" in error_msg or "pdf2image" in error_msg):
                    st.error(
                        " Error: Missing OCR dependencies.\n\n"
                        "Please install the required packages:\n"
                        "`pip install pytesseract pdf2image Pillow`"
                    )
                else:
                    st.error(f" Error processing document: {error_msg}")

# Document summary section (only shown after upload)
if st.session_state.document_uploaded and st.session_state.summary:
    st.header("📝 Document Summary")
    st.write(st.session_state.summary)
      # Mode selection
    if st.session_state.mode is None:
        st.header("Choose Interaction Mode")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💬 Ask Anything", use_container_width=True):
                st.session_state.mode = "ask"
                st.session_state["rerun"] = True
        
        with col2:
            if st.button("🧩 Challenge Me", use_container_width=True):
                st.session_state.mode = "challenge"
                st.session_state["rerun"] = True

# Ask Anything Mode
if st.session_state.document_uploaded and st.session_state.mode == "ask":
    st.header("💬 Ask Anything Mode")
    st.markdown("""
    Ask any question about the document. The assistant will provide an answer based on the document content.
    """)
    
    # Question input
    question = st.text_input("Ask a question about the document:")
    use_context = st.checkbox("Use conversation context for follow-up questions", value=True)
    
    if question and st.button("Submit Question"):
        with st.spinner("Generating answer..."):
            try:
                # Get answer from API
                result = ask_question(question, with_context=use_context)
                
                # Add to history
                st.session_state.qa_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "source": result.get("source")
                })
            
            except Exception as e:
                st.error(f"Error getting answer: {str(e)}")
    
    # Display QA history
    if st.session_state.qa_history:
        st.subheader("Questions & Answers")
        
        for i, qa in enumerate(st.session_state.qa_history):
            with st.expander(f"Q: {qa['question']}", expanded=(i == len(st.session_state.qa_history) - 1)):
                st.markdown(f"**Answer:**\n{qa['answer']}")
                
                if qa.get("source"):
                    st.markdown("**Source Text:**")
                    st.info(qa["source"])
    
    # Option to go back
    if st.button("Back to Mode Selection"):
        st.session_state.mode = None
        st.session_state["rerun"] = True

# Challenge Mode
if st.session_state.document_uploaded and st.session_state.mode == "challenge":
    st.header("🧩 Challenge Me Mode")
    st.markdown("""
    Test your understanding of the document with these AI-generated challenges.
    Answer the questions and get feedback on your responses.
    """)
    
    if st.session_state.challenges:
        for i, challenge in enumerate(st.session_state.challenges):
            st.subheader(f"Challenge {i + 1}")
            st.write(challenge["question"])

            # Keys for widgets
            answer_key = f"answer_{i}"
            submit_key = f"submit_{i}"

            # Input area
            answer = st.text_area(f"Your answer for Challenge {i + 1}:", key=answer_key)

            # Submit button (first)
            if st.button(f"Submit Answer {i + 1}", key=submit_key):
                if answer.strip():
                    with st.spinner("Evaluating your answer..."):
                        try:
                            evaluation = evaluate_challenge_answer(i, answer.strip())
                            st.session_state.challenge_answers[i] = answer
                            st.session_state.challenge_evaluations[i] = evaluation
                        except Exception as e:
                            st.error(f"Error evaluating answer: {str(e)}")
                else:
                    st.warning("Please enter an answer before submitting.")

            # Then display evaluation, if it exists
            if i in st.session_state.challenge_evaluations:
                evaluation = st.session_state.challenge_evaluations[i]
                with st.expander("Debug Evaluation JSON"):
                    st.json(evaluation)

                correct_value = str(evaluation.get("correct", "")).strip().lower()

                if "yes" in correct_value or "true" in correct_value or "correct" == correct_value:
                    st.success(f"Correct. Score: {evaluation.get('score', 100)}/100")
                elif "partial" in correct_value:
                    st.warning(f"Partially correct. Score: {evaluation.get('score', 60)}/100")
                elif "no" in correct_value or "false" in correct_value or "incorrect" == correct_value:
                    st.error(f"Incorrect. Score: {evaluation.get('score', 0)}/100")
                else:
                    st.error(f"Unable to determine result. Raw correct value: '{correct_value}'. Score: {evaluation.get('score', 'N/A')}/100")


                st.markdown(f"**Feedback:** {evaluation.get('feedback', 'No feedback available.')}")
                with st.expander("See Justification"):
                    st.markdown(evaluation.get("justification", "No justification provided."))

            st.divider()

    
    else:
        st.warning("No challenges available. Please try uploading the document again.")
    
    # Option to go back
    if st.button("Back to Mode Selection"):
        st.session_state.mode = None
        st.session_state["rerun"] = True

# Reset button (at the bottom)
if st.session_state.document_uploaded:
    if st.button("Reset & Upload New Document"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.session_state["rerun"] = True

# Footer
st.markdown("""
---
### About Smart Research Assistant

This tool helps you analyze and interact with research documents:
- **Document Upload**: Upload PDF or TXT files
- **Auto-Summary**: Get a concise summary of the document
- **Ask Anything**: Ask questions about the document content
- **Challenge Me**: Test your understanding with AI-generated challenges

All answers and feedback are grounded in the document content to ensure accuracy.
""")
