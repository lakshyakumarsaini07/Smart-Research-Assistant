# Smart Research Assistant

A document-aware AI assistant for research summarization and comprehension powered by Google's Gemini 2.5 Flash API. Upload a document (PDF or TXT), get a concise summary, ask questions about the content, or test your understanding with AI-generated challenges.

## Features

### Document Upload

- Upload PDF (text-based or scanned) or TXT files
- Automatic text extraction and processing
- Built with PyMuPDF, pdfplumber, and OCR capabilities
- Supports image-based PDFs with OCR (requires additional setup)

### Auto-Summary

- Automatically generates a concise summary (≤150 words)
- Captures key points from the document
- Powered by Google's Gemini 2.5 Flash LLM

### Ask Anything Mode

- Ask any question about the document
- Get answers grounded in the document content
- Includes justification with source references
- Utilizes retrieval-augmented generation (RAG)

### Challenge Me Mode

- AI-generated comprehension challenges
- Automatic evaluation of your answers
- Detailed feedback and scoring
- References to supporting document content

### Contextual Grounding

- All answers and feedback are strictly based on document content
- Source text provided for transparency
- Memory for conversation context in Ask Anything mode

## Architecture

### Backend (FastAPI)

- Document processing and text extraction
- Vector embedding and storage (FAISS)
- Question answering with LangChain
- Challenge generation and evaluation

### Frontend (Streamlit)

- Clean, intuitive user interface
- Document upload and display
- Interactive Q&A interface
- Challenge response system

### NLP Components

- Together AI API models for natural language understanding
- Document chunking and embedding
- Retrieval-augmented generation for accurate responses
- Contextual memory for follow-up questions

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Google Gemini API key

### Installation

1. Clone the repository

   ```bash
   git clone https://github.com/yourusername/smart-research-assistant.git
   cd smart-research-assistant
   ```

2. Set up a virtual environment

   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies

   ```bash
   pip install -e .
   ```

4. (Optional) Install OCR dependencies for scanned documents

   This step is only needed if you want to process scanned PDFs or image-based documents:

   ```bash
   # Install Python packages
   pip install pytesseract pdf2image Pillow

   # Install Tesseract OCR (platform-specific)
   # Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   # macOS: brew install tesseract
   # Linux (Debian/Ubuntu): sudo apt install tesseract-ocr
   ```

   You can also use the provided installation scripts:

   ```bash
   # On Windows
   install_ocr_windows.bat

   # On macOS/Linux
   ./install_ocr_unix.sh
   ```

5. Set your Gemini API key

   ```bash
   # On Windows (PowerShell)
   $env:GEMINI_API_KEY = "your-api-key"
   # On macOS/Linux
   export GEMINI_API_KEY="your-api-key"
   ```

   Alternatively, create a `.env` file in the project root:

   ```
   GEMINI_API_KEY=your-api-key
   ```

## Running the Application

1. Start the application

   ```bash
   python main.py
   ```

   Alternatively, on Windows:

   ```bash
   run.bat
   ```

   Or on Linux/macOS:

   ```bash
   ./run.sh
   ```

2. The application will start both the FastAPI backend and Streamlit frontend
   - FastAPI server: http://localhost:8000
   - Streamlit interface: http://localhost:8501 (opens automatically)

## Usage

1. **Upload a Document**:

   - Click on the file uploader and select a PDF or TXT file
   - Click "Process Document" to analyze the document

2. **Review the Summary**:

   - The system will automatically generate a summary
   - Review the key points from the document

3. **Choose Interaction Mode**:

   - **Ask Anything**: Ask questions about the document
   - **Challenge Me**: Test your comprehension with AI challenges

4. **Ask Questions**:

   - Type your question in the text input
   - View the answer with source references
   - Ask follow-up questions using conversation context

5. **Complete Challenges**:
   - Answer the AI-generated questions
   - Get immediate feedback and scoring
   - See justifications for correct answers

## Example Use Cases

- **Academic Research**: Quickly understand academic papers and extract key information
- **Legal Document Analysis**: Parse and query complex legal documents
- **Business Reports**: Extract insights from lengthy business reports
- **Educational Content**: Create interactive learning experiences from textbooks or articles
- **Technical Documentation**: Navigate and query complex technical documentation

## Technical Details

- **Text Extraction**: PyMuPDF, pdfplumber, PyPDF, Tesseract OCR
- **Embedding & Vector Storage**: HuggingFace Embeddings (all-MiniLM-L6-v2), FAISS
- **Language Models**: Google Gemini 2.5 Flash
- **Document QA**: LangChain with RetrievalQA
- **Frontend**: Streamlit
- **API**: FastAPI

## Troubleshooting

### No text could be extracted from the document

If you encounter this error when uploading a PDF:

1. **For scanned/image-based PDFs**:

   - Install OCR dependencies: `pip install pytesseract pdf2image Pillow`
   - Install Tesseract OCR (see installation instructions above)
   - Make sure Tesseract is in your system PATH

2. **For encrypted/protected PDFs**:

   - Remove password protection or content restrictions before uploading
   - Save a copy of the PDF with reduced security settings

3. **For other issues**:
   - Try converting the PDF to a different format and back
   - Use a different PDF viewer to save a copy (sometimes resolves encoding issues)
   - Extract the text manually to a TXT file and upload that instead

## License

[MIT License](LICENSE)
