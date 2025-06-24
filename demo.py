"""
Demonstration of the Smart Research Assistant capabilities with Gemini
"""
import os
import sys
import asyncio
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

async def demo_llm():
    """Demonstrate LLM capabilities"""
    from app.utils.llm_interface import GeminiLLM
    
    print("\n=== LLM Demo (Gemini 2.5 Flash) ===")
    llm = GeminiLLM(temperature=0.7)
    
    # Simple question answering
    questions = [
        "What is retrieval-augmented generation (RAG)?",
        "How can AI assist with research summarization?",
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        response = llm.generate(question)
        print(f"A: {response.strip()}")

async def demo_document_processing():
    """Demonstrate document processing capabilities"""
    from app.modules.document_loader import process_document, create_document_chunks
    from app.modules.summarizer import summarize_document
    from app.utils.config import CHUNK_SIZE, CHUNK_OVERLAP
    
    print("\n=== Document Processing Demo ===")
    
    # Use a sample document path - replace with an actual file path for testing
    sample_file_path = "README.md"
    
    if os.path.exists(sample_file_path):
        print(f"Processing document: {sample_file_path}")
        
        # Process the document
        document_text = process_document(sample_file_path, ".md")
        print(f"Document loaded: {len(document_text)} characters")
        
        # Create document chunks
        document_chunks = create_document_chunks(document_text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"Document split into {len(document_chunks)} chunks")
        
        # Generate a summary
        summary = summarize_document(document_chunks)
        print("\nDocument Summary:")
        print(summary)
    else:
        print(f"Sample document not found: {sample_file_path}")

async def main():
    """Run demo functions"""
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables")
        sys.exit(1)
    
    print("="*50)
    print("🧠 Smart Research Assistant (Gemini 2.5 Flash Edition)")
    print("="*50)
    
    # Run demos
    await demo_llm()
    await demo_document_processing()

if __name__ == "__main__":
    asyncio.run(main())
