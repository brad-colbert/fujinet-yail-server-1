#!/usr/bin/env python3
"""
YAIL Image Generation Module

This module contains the image generation functionality for the YAIL server,
including support for OpenAI's DALL-E and Google's Gemini models.
"""

import os
import time
import logging
import base64
import traceback
from io import BytesIO
from typing import Optional, Dict, Any, Union
from PIL import Image
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

# For OpenAI API
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Install with: pip install openai")

# For Google Gemini API
try:
    import google.generativeai as genai
    from google.generativeai import types as genai_types
    GEMINI_AVAILABLE = True
    # Configure the Gemini API with API key if provided
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key:
        logger.info(f"Gemini API key found in environment, configuring Gemini API")
        genai.configure(api_key=gemini_api_key)
    else:
        logger.info("No GEMINI_API_KEY found in environment. Using default authentication.")
except ImportError:
    GEMINI_AVAILABLE = False
    logger.error("Google Generative AI library not available. Install with: pip install google-generativeai")


class ImageGenConfig:
    """
    Configuration class for image generation.
    Provides validation and management of image generation parameters.
    """
    
    # Valid model prefixes for determining which API to use
    OPENAI_MODEL_PREFIXES = ["dall-e-", "gpt-"]
    GEMINI_MODEL_PREFIXES = ["gemini"]
    
    # Default values for configuration
    DEFAULT_MODEL = "dall-e-3"
    DEFAULT_SIZE = "1024x1024"
    DEFAULT_QUALITY = "standard"
    DEFAULT_STYLE = "vivid"
    
    # Valid options for OpenAI configuration
    VALID_SIZES = ["1024x1024", "1792x1024", "1024x1792"]
    VALID_QUALITIES = ["standard", "hd"]
    VALID_STYLES = ["vivid", "natural"]
    
    def __init__(self):
        # Load settings from environment variables
        self.model = os.environ.get("GEN_MODEL", os.environ.get("OPENAI_MODEL", self.DEFAULT_MODEL))
        
        # Force model to be gemini-2.5-pro-exp-03-25 if it contains "gemini" in the name
        if "gemini" in self.model.lower():
            logger.info(f"Model name contains 'gemini', ensuring it's treated as a Gemini model: {self.model}")
            # Make sure we're using the full model name for Gemini
            if self.model.lower() == "gemini":
                self.model = "gemini-2.5-pro-exp-03-25"
                logger.info(f"Updated generic 'gemini' to specific model: {self.model}")
        
        self.size = os.environ.get("OPENAI_SIZE", self.DEFAULT_SIZE)
        self.quality = os.environ.get("OPENAI_QUALITY", self.DEFAULT_QUALITY)
        self.style = os.environ.get("OPENAI_STYLE", self.DEFAULT_STYLE)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        self.system_prompt = os.environ.get("OPENAI_SYSTEM_PROMPT", "You are an image generation assistant. Generate an image based on the user's description.")
        
        # Debug: Print loaded configuration
        logger.info(f"ImageGenConfig initialized with model: {self.model}")
        logger.info(f"Environment variables: GEN_MODEL={os.environ.get('GEN_MODEL', 'not set')}, OPENAI_MODEL={os.environ.get('OPENAI_MODEL', 'not set')}")
        logger.info(f"OPENAI_API_KEY: {'Set' if self.api_key else 'Not set'}")
        logger.info(f"GEMINI_API_KEY: {'Set' if self.gemini_api_key else 'Not set'}")
        
        # Validate the loaded settings
        if not self.is_valid_model(self.model):
            logger.warning(f"Unknown model format: {self.model}. Using default: {self.DEFAULT_MODEL}")
            self.model = self.DEFAULT_MODEL
            
        if self.size not in self.VALID_SIZES:
            logger.warning(f"Invalid OPENAI_SIZE in environment: {self.size}. Using default: {self.DEFAULT_SIZE}")
            self.size = self.DEFAULT_SIZE
            
        if self.quality not in self.VALID_QUALITIES:
            logger.warning(f"Invalid OPENAI_QUALITY in environment: {self.quality}. Using default: {self.DEFAULT_QUALITY}")
            self.quality = self.DEFAULT_QUALITY
            
        if self.style not in self.VALID_STYLES:
            logger.warning(f"Invalid OPENAI_STYLE in environment: {self.style}. Using default: {self.DEFAULT_STYLE}")
            self.style = self.DEFAULT_STYLE
    
    def is_valid_model(self, model: str) -> bool:
        """
        Check if the model name is valid by checking if it starts with a recognized prefix.
        
        Args:
            model (str): The model name to check
            
        Returns:
            bool: True if the model is valid, False otherwise
        """
        if not model:
            return False
            
        # Check if the model starts with any of the recognized prefixes
        for prefix in self.OPENAI_MODEL_PREFIXES + self.GEMINI_MODEL_PREFIXES:
            if model.lower().startswith(prefix.lower()):
                return True
                
        return False
    
    def is_openai_model(self, model: str = None) -> bool:
        """
        Check if the model is an OpenAI model.
        
        Args:
            model (str, optional): The model name to check. If None, uses the configured model.
            
        Returns:
            bool: True if the model is an OpenAI model, False otherwise
        """
        model = model or self.model
        
        for prefix in self.OPENAI_MODEL_PREFIXES:
            if model.lower().startswith(prefix.lower()):
                return True
                
        return False
    
    def is_gemini_model(self, model: str = None) -> bool:
        """
        Check if the model is a Gemini model.
        
        Args:
            model (str, optional): The model name to check. If None, uses the configured model.
            
        Returns:
            bool: True if the model is a Gemini model, False otherwise
        """
        model = model or self.model
        
        # First check with the prefix method
        for prefix in self.GEMINI_MODEL_PREFIXES:
            if model.lower().startswith(prefix.lower()):
                return True
        
        # Then check if "gemini" appears anywhere in the model name
        if "gemini" in model.lower():
            logger.info(f"Model name contains 'gemini', treating as Gemini model: {model}")
            return True
                
        return False
    
    def set_model(self, model: str) -> bool:
        """
        Set the model if valid, otherwise return False
        """
        if self.is_valid_model(model):
            self.model = model
            logger.info(f"Model set to: {model}")
            return True
        else:
            logger.warning(f"Invalid model: {model}. Model must start with one of: {', '.join(self.OPENAI_MODEL_PREFIXES + self.GEMINI_MODEL_PREFIXES)}")
            return False
    
    def set_size(self, size: str) -> bool:
        """
        Set the size if valid, otherwise return False
        """
        if size in self.VALID_SIZES:
            self.size = size
            logger.info(f"Size set to: {size}")
            return True
        else:
            logger.warning(f"Invalid size: {size}. Valid options are: {', '.join(self.VALID_SIZES)}")
            return False
    
    def set_quality(self, quality: str) -> bool:
        """
        Set the quality if valid, otherwise return False
        """
        if quality in self.VALID_QUALITIES:
            self.quality = quality
            logger.info(f"Quality set to: {quality}")
            return True
        else:
            logger.warning(f"Invalid quality: {quality}. Valid options are: {', '.join(self.VALID_QUALITIES)}")
            return False
    
    def set_style(self, style: str) -> bool:
        """
        Set the style if valid, otherwise return False
        """
        if style in self.VALID_STYLES:
            self.style = style
            logger.info(f"Style set to: {style}")
            return True
        else:
            logger.warning(f"Invalid style: {style}. Valid options are: {', '.join(self.VALID_STYLES)}")
            return False
    
    def set_api_key(self, api_key: str) -> None:
        """
        Set the API key
        """
        self.api_key = api_key
        logger.info("API key updated")
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Set the system prompt
        """
        self.system_prompt = system_prompt
        logger.info(f"System prompt updated: {system_prompt}")
    
    def __str__(self) -> str:
        """
        String representation of the configuration
        """
        return f"ImageGenConfig(model={self.model}, size={self.size}, quality={self.quality}, style={self.style})"


def generate_image_with_openai(prompt: str, api_key: str = None, model: str = None, 
                               size: str = None, quality: str = None, style: str = None) -> Optional[str]:
    """
    Generate an image using OpenAI's image generation models and return the URL.
    
    Args:
        prompt (str): The text prompt to generate an image from
        api_key (str, optional): OpenAI API key. If None, uses OPENAI_API_KEY environment variable
        model (str, optional): The model to use. If None, uses the configured model.
        size (str, optional): Image size. Options for DALL-E 3: "1024x1024" (default), "1792x1024", or "1024x1792"
        quality (str, optional): Image quality. Options: "standard" (default) or "hd" (DALL-E 3 only)
        style (str, optional): Image style. Options: "vivid" (default) or "natural" (DALL-E 3 only)
        
    Returns:
        str: URL of the generated image or None if generation failed
    """
    if not OPENAI_AVAILABLE:
        logger.error("OpenAI library not available. Install with: pip install openai")
        return None
    
    # Use the global config if parameters are not provided
    global gen_config
    
    if api_key is None:
        api_key = gen_config.api_key
    
    if model is None:
        model = gen_config.model
    
    if size is None:
        size = gen_config.size
    
    if quality is None:
        quality = gen_config.quality
    
    if style is None:
        style = gen_config.style
    
    if not api_key:
        logger.error("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        return None
    
    try:
        logger.info(f"Generating image with OpenAI model: {model}, prompt: '{prompt}'")
        
        # Initialize the OpenAI client with the API key
        client = openai.OpenAI(api_key=api_key)
        
        # Generate image with OpenAI
        # For DALL-E models, we need to check if the model name starts with "dall-e"
        if model.lower().startswith("dall-e"):
            # For DALL-E 3, we can use quality and style parameters
            if model.lower() == "dall-e-3":
                response = client.images.generate(
                    model=model,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    style=style,
                    n=1,
                    response_format="url"
                )
            else:
                # For DALL-E 2 and other versions, don't use quality and style
                response = client.images.generate(
                    model=model,
                    prompt=prompt,
                    size=size,
                    n=1,
                    response_format="url"
                )
        else:
            # For other OpenAI models, try to use the standard parameters
            # This is a fallback for future models or other OpenAI models
            logger.warning(f"Using non-DALL-E model: {model}. Some parameters may be ignored.")
            response = client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                n=1,
                response_format="url"
            )
            
        # Extract the URL from the response
        image_url = response.data[0].url
        logger.info(f"Image generated successfully with OpenAI: {image_url}")
        return image_url
            
    except Exception as e:
        logger.error(f"Error generating image with OpenAI: {e}")
        # Print full exception traceback for debugging
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def generate_image_with_gemini(prompt: str) -> Optional[str]:
    """
    Generate an image using Google's Gemini API and return the URL or path to the saved image.
    
    Args:
        prompt (str): The text prompt to generate an image from
        
    Returns:
        str: Path to the saved image or None if generation failed
    """
    if not GEMINI_AVAILABLE:
        logger.error("Google Generative AI library not available. Install with: pip install google-generativeai")
        return None
    
    try:
        # Get the exact model name from the configuration
        model_name = gen_config.model
        logger.info(f"Generating image with Gemini model: {model_name}, prompt: '{prompt}'")
        
        # Check if we have an API key
        api_key = gen_config.gemini_api_key
        if not api_key:
            logger.error("Gemini API key not provided. Set GEMINI_API_KEY environment variable.")
            return None
            
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Generate image with Gemini
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(contents=prompt)
        
        # Extract and save the image
        image_saved = False
        image_path = None
        
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    # Save the image to a file
                    image_data = base64.b64decode(part.inline_data.data)
                    image = Image.open(BytesIO(image_data))
                    
                    # Create a directory for generated images if it doesn't exist
                    os.makedirs('generated_images', exist_ok=True)
                    
                    # Generate a unique filename based on timestamp
                    timestamp = int(time.time())
                    image_path = f"generated_images/gemini-{timestamp}.png"
                    
                    # Save the image
                    image.save(image_path)
                    image_saved = True
                    
                    logger.info(f"Image generated successfully with Gemini: {image_path}")
                    break
                elif hasattr(part, 'text') and part.text:
                    logger.info(f"Gemini text response: {part.text}")
        else:
            logger.error("No candidates in Gemini response")
        
        if image_saved and image_path:
            # Return the absolute path to the saved image
            abs_path = os.path.abspath(image_path)
            return abs_path
        else:
            logger.error("Failed to extract image from Gemini response")
            return None
            
    except Exception as e:
        logger.error(f"Error generating image with Gemini: {e}")
        # Print full exception traceback for debugging
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def generate_image(prompt: str, model: str = None) -> Optional[str]:
    """
    Generate an image using the specified model and return the URL or path.
    
    Args:
        prompt (str): The text prompt to generate an image from
        model (str, optional): The model to use. If None, uses the configured model.
        
    Returns:
        str: URL or path to the generated image or None if generation failed
    """
    global gen_config
    
    # Use the global config if model is not provided
    if model is None:
        model = gen_config.model
    
    logger.info(f"Generating image with model: {model}, prompt: '{prompt}'")
    
    # Debug the model detection
    logger.info(f"Model detection: is_gemini={gen_config.is_gemini_model(model)}, is_openai={gen_config.is_openai_model(model)}")
    
    # Generate image based on the model
    if gen_config.is_gemini_model(model):
        logger.info(f"Using Gemini API for model: {model}")
        return generate_image_with_gemini(prompt)
    elif gen_config.is_openai_model(model):
        logger.info(f"Using OpenAI API for model: {model}")
        return generate_image_with_openai(prompt, model=model)
    else:
        # If model detection fails, try to infer from the model name
        if "gemini" in model.lower():
            logger.info(f"Model name contains 'gemini', using Gemini API for model: {model}")
            return generate_image_with_gemini(prompt)
        elif any(prefix in model.lower() for prefix in ["dall-e", "gpt"]):
            logger.info(f"Model name contains OpenAI prefix, using OpenAI API for model: {model}")
            return generate_image_with_openai(prompt, model=model)
        else:
            logger.error(f"Unsupported model: {model}")
            return None


# Create a global instance of ImageGenConfig - will be initialized later
gen_config = None

def initialize_gen_config():
    """Initialize the global gen_config instance"""
    global gen_config
    gen_config = ImageGenConfig()
    return gen_config
