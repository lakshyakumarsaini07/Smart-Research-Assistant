"""
Interface for Google Gemini API LLM
"""
import os
import json
import requests
from typing import Optional, List, Dict, Any

from app.utils.config import GEMINI_API_KEY, GEMINI_API_URL, DEFAULT_MODEL, EMBEDDING_MODEL

class GeminiLLM:
    """
    Class for interfacing with the Google Gemini API
    """
    
    def __init__(self, 
                 model: str = DEFAULT_MODEL, 
                 temperature: float = 0.7, 
                 max_tokens: int = 1000,
                 api_key: Optional[str] = None,
                 api_url: Optional[str] = None):
        """
        Initialize the Gemini API LLM interface
        
        Args:
            model: Model to use for generation
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            api_key: Gemini API key (default: from environment)
            api_url: Gemini API URL (default: from config)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or GEMINI_API_KEY
        self.api_url = api_url or GEMINI_API_URL
        
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
    
    def generate(self, prompt: str, context: str = "", verbose: bool = False) -> Optional[str]:
        """
        Generate a response using the Gemini API.
        
        Args:
            prompt: The main prompt to generate a response for
            context: Additional context for the prompt
            verbose: Whether to print the full API response
            
        Returns:
            Generated response text or None if an error occurs
        """
        try:
            # Construct the complete prompt with context
            full_prompt = f"{context}\n{prompt}" if context else prompt
            
            # Construct the request URL with API key
            url = f"{self.api_url}/{self.model}:generateContent?key={self.api_key}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": full_prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens,
                    "topP": 0.7,
                    "topK": 50
                }
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            if verbose:
                print(f"API Response: {json.dumps(result, indent=2)}")
            
            # Extract the generated text from the response
            if ('candidates' in result and 
                len(result['candidates']) > 0 and 
                'content' in result['candidates'][0] and 
                'parts' in result['candidates'][0]['content'] and 
                len(result['candidates'][0]['content']['parts']) > 0):
                
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                if verbose:
                    print(f"Unexpected response structure: {result}")
                return None
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

class GeminiEmbeddings:
    """
    Class for generating embeddings using Gemini API
    """
    
    def __init__(self, 
                 model: str = EMBEDDING_MODEL,
                 api_key: Optional[str] = None,
                 api_url: Optional[str] = None):
        """
        Initialize the Gemini API embeddings interface
        
        Args:
            model: Model to use for embeddings
            api_key: Gemini API key (default: from environment)
            api_url: Gemini API URL for embeddings (default: from config)
        """
        self.model = model
        self.api_key = api_key or GEMINI_API_KEY
        self.api_url = "https://generativelanguage.googleapis.com/v1beta"
        
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Process each text individually and collect results
            embeddings = []
            for text in texts:
                embedding = self.embed_query(text)
                if embedding:
                    embeddings.append(embedding)
            return embeddings
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return []
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            url = f"{self.api_url}/{self.model}:embedText?key={self.api_key}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "text": text
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            if 'embedding' in result and 'values' in result['embedding']:
                return result['embedding']['values']
            else:
                print(f"Unexpected response structure: {result}")
                return []
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
