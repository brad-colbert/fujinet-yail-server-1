#!/usr/bin/env python3
"""
Test script for model detection in YAIL server
"""
import os
import logging
from yail_gen import ImageGenConfig

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_detection():
    """Test the model detection logic"""
    # Set environment variables for testing
    os.environ["GEN_MODEL"] = "gemini-2.5-pro-exp-03-25"
    os.environ["OPENAI_API_KEY"] = "test_openai_key"
    os.environ["GEMINI_API_KEY"] = "test_gemini_key"
    
    # Create config instance
    config = ImageGenConfig()
    
    # Print config details
    logger.info(f"Model from config: {config.model}")
    logger.info(f"Is Gemini model: {config.is_gemini_model()}")
    logger.info(f"Is OpenAI model: {config.is_openai_model()}")
    
    # Test with different model names
    test_models = [
        "gemini-2.5-pro-exp-03-25",
        "gemini",
        "dall-e-3",
        "dall-e-2",
        "gpt-4o",
        "invalid-model"
    ]
    
    for model in test_models:
        logger.info(f"\nTesting model: {model}")
        logger.info(f"Is Gemini model: {config.is_gemini_model(model)}")
        logger.info(f"Is OpenAI model: {config.is_openai_model(model)}")

if __name__ == "__main__":
    test_model_detection()
