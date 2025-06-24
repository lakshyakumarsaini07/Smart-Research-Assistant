"""
Configuration utilities for the Smart Research Assistant
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "models/embedding-001"

# Document processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector database settings
VECTOR_DB_PATH = "vector_db"

# Summary settings
MAX_SUMMARY_WORDS = 150

# Challenge settings
NUM_CHALLENGES = 3

# Server settings
FASTAPI_PORT = 8000
STREAMLIT_PORT = 8501
