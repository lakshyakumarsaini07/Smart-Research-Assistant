"""
Document summarization utilities
"""
from typing import List, Any, Dict, Optional, Sequence

from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation, LLMResult

from app.utils.config import MAX_SUMMARY_WORDS
from app.utils.llm_interface import GeminiLLM

class LangchainGeminiLLM(LLM):
    """
    Wrapper class for GeminiLLM to make it compatible with Langchain
    """
    
    # Define all class fields for pydantic
    temperature: float = 0.0
    gemini_llm: Optional[Any] = None
    
    def __init__(self, temperature: float = 0.0, **kwargs: Any):
        """
        Initialize the wrapper for Gemini LLM
        
        Args:
            temperature: Sampling temperature
        """        
        super().__init__(temperature=temperature, **kwargs)
        self.gemini_llm = GeminiLLM(temperature=temperature)
    
    def _call(self, prompt: str, stop=None) -> str:
        """
        Generate a response from the LLM
        
        Args:
            prompt: Input prompt
            stop: Stop tokens (not used)
            
        Returns:
            Generated text
        """
        response = self.gemini_llm.generate(prompt)
        return response if response else ""
    
    def _generate(
        self,
        prompts: Sequence[str],
        stop: Optional[Sequence[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        Generate text based on prompts and stops.
        
        Args:
            prompts: List of prompts to generate text from
            stop: List of stop sequences
            run_manager: Callback manager
            **kwargs: Additional parameters
            
        Returns:
            LLMResult containing generated text
        """
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop)
            generations.append([Generation(text=text)])
        
        return LLMResult(generations=generations)
    
    @property
    def _llm_type(self) -> str:
        """Type of LLM"""
        return "gemini_llm"

def create_summarizer() -> LangchainGeminiLLM:
    """
    Create an instance of the summarizer model
    
    Returns:
        LangchainGeminiLLM model instance
    """
    # Create GeminiLLM with lower temperature for more factual summaries
    return LangchainGeminiLLM(temperature=0.3)

def summarize_document(documents: List[Document]) -> str:
    """
    Generate a summary of the document
    
    Args:
        documents: List of document chunks
        
    Returns:
        Summary text
    """
    llm = create_summarizer()
    
    # Create the summarization prompt
    prompt_template = f"""
    Your task is to create a comprehensive yet concise summary of the following document.
    Focus on the main topics, key points, and important details.
    Make the summary informative and well-structured, with clear paragraphs.
    Use bullet points for lists of items when appropriate.
    Keep the summary under {MAX_SUMMARY_WORDS} words.
    
    DOCUMENT:
    {{text}}
    
    SUMMARY:
    """
    
    # Create the summarization chain
    summary_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(
        llm=llm,
        chain_type="stuff",
        prompt=summary_prompt
    )
    
    # Run the chain
    result = chain.invoke(documents)
    
    # Return the generated summary
    return result.get("output_text", "").strip()
