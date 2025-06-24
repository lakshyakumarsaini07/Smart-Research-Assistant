"""
Main entry point for the Smart Research Assistant application.
This file initializes and runs both the FastAPI backend and Streamlit frontend.
"""
import os
import sys
import subprocess
import threading
import time
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_api_key():
    """Check if Gemini API key is set in environment variables"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables")
        api_key = input("Please enter your Gemini API key: ")
        os.environ["GEMINI_API_KEY"] = api_key
        # Save to .env file
        with open(".env", "a") as env_file:
            env_file.write(f"\nGEMINI_API_KEY={api_key}\n")
        print("API key saved to .env file")
    return api_key

def run_fastapi_server():
    """Run the FastAPI server"""
    print("Starting FastAPI server...")
    subprocess.run(
        ["uvicorn", "app.core.api:app", "--host", "0.0.0.0", "--port", "8000"],
        check=True
    )

def run_streamlit_server():
    """Run the Streamlit server"""
    print("Starting Streamlit server...")
    subprocess.run(
        ["streamlit", "run", "app/frontend/streamlit_app.py", "--server.port", "8501"],
        check=True
    )

async def test_gemini_api():
    """Test the Gemini API with a simple request"""
    from app.utils.llm_interface import GeminiLLM
    
    print("Testing Gemini API connection...")
    try:
        llm = GeminiLLM(temperature=0.7)
        response = llm.generate("Hello, this is a test of the Gemini API. Please respond with a short greeting.")
        
        if response:
            print("✅ Successfully connected to Gemini API")
            print(f"API Response: {response.strip()}")
            return True
        else:
            print("❌ Failed to connect to Gemini API")
            print("Please check your API key and internet connection")
            return False
    except Exception as e:
        print(f"❌ Error testing Gemini API: {e}")
        return False

def main():
    """Main entry point"""
    print("="*50)
    print("🧠 Smart Research Assistant (Gemini 2.5 Flash Edition)")
    print("A document-aware AI assistant for research summarization and comprehension")
    print("="*50)
    
    # Check if API key is set
    check_api_key()
    
    # Test Gemini API connection
    api_test_success = asyncio.run(test_gemini_api())
    if not api_test_success:
        print("Would you like to continue anyway? (y/n)")
        choice = input("> ").strip().lower()
        if choice != 'y':
            print("Exiting application. Please check your API key and try again.")
            sys.exit(1)
    
    # Test HuggingFace embeddings (without blocking if it fails)
    try:
        from app.modules.vector_store import VectorStore
        vector_store = VectorStore()
        test_text = "Testing embeddings functionality"
        embedding = vector_store.embeddings.embed_query(test_text)
        if isinstance(embedding, list) and len(embedding) > 0:
            print("✅ Successfully tested HuggingFace embeddings")
        else:
            print("⚠️ Warning: HuggingFace embeddings test returned unexpected result")
    except Exception as e:
        print(f"⚠️ Warning: Could not test HuggingFace embeddings: {e}")
        print("The application will continue, but document search functionality might be affected.")
    
    # Start FastAPI server in a separate thread
    api_thread = threading.Thread(target=run_fastapi_server)
    api_thread.daemon = True
    api_thread.start()
    
    # Wait for FastAPI to start
    print("Waiting for FastAPI server to start...")
    time.sleep(3)
    
    # Start Streamlit server in the main thread
    run_streamlit_server()

if __name__ == "__main__":
    main()