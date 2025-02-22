# model.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import spacy
import re
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CVScorer:
    def __init__(self):
        """Initialize the CV scoring system with a lighter model"""
        try:
            # Use a smaller, faster model
            self.model_name = "facebook/opt-125m"  # Much smaller than Mistral-7B
            
            # Load model with optimizations
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Use half precision
                low_cpu_mem_usage=True,
            )
            
            # Create pipeline with optimized settings
            self.scoring_model = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
                model_kwargs={"torch_dtype": torch.float16}
            )
            
            # Load smaller spaCy model
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            logger.info("CV Scorer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CV Scorer: {str(e)}")
            raise

    def score_cv(self, cv_text: str) -> float:
        """Simplified scoring function with timeout protection"""
        try:
            # Simple keyword matching for faster processing
            keywords = {
                'ml_skills': ['machine learning', 'deep learning', 'neural networks', 'ai'],
                'python_skills': ['python', 'tensorflow', 'pytorch', 'keras'],
                'research': ['research', 'paper', 'publication', 'journal'],
                'math': ['statistics', 'mathematics', 'algorithms']
            }
            
            # Calculate score based on keyword presence
            scores = []
            cv_text_lower = cv_text.lower()
            
            for category, terms in keywords.items():
                category_score = sum(1 for term in terms if term in cv_text_lower)
                scores.append(min(category_score * 20, 100))  # Cap at 100
            
            final_score = sum(scores) / len(scores)
            return min(final_score, 100)  # Ensure score doesn't exceed 100
            
        except Exception as e:
            logger.error(f"Error in CV scoring: {str(e)}")
            return 0.0

# Initialize singleton instance
cv_scorer = CVScorer()