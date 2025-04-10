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
    
    VALID_MODELS = ["dall-e-3", "dall-e-2", "gemini"]
    VALID_SIZES = ["1024x1024", "1792x1024", "1024x1792"]
    VALID_QUALITIES = ["standard", "hd"]
    VALID_STYLES = ["vivid", "natural"]
    
    def __init__(self):
        # Default settings
        self.model = os.environ.get("GEN_MODEL", os.environ.get("OPENAI_MODEL", "dall-e-3"))
        self.size = os.environ.get("OPENAI_SIZE", "1024x1024")
        self.quality = os.environ.get("OPENAI_QUALITY", "standard")
        self.style = os.environ.get("OPENAI_STYLE", "vivid")
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.system_prompt = os.environ.get("OPENAI_SYSTEM_PROMPT", "You are an image generation assistant. Generate an image based on the user's description.")
        
        # Debug: Print loaded configuration
        logger.info(f"ImageGenConfig initialized with model: {self.model}")
        
        # Validate the loaded settings
        if self.model not in self.VALID_MODELS:
            logger.warning(f"Invalid GEN_MODEL in environment: {self.model}. Using default: dall-e-3")
            self.model = "dall-e-3"
            
        if self.size not in self.VALID_SIZES:
            logger.warning(f"Invalid OPENAI_SIZE in environment: {self.size}. Using default: 1024x1024")
            self.size = "1024x1024"
            
        if self.quality not in self.VALID_QUALITIES:
            logger.warning(f"Invalid OPENAI_QUALITY in environment: {self.quality}. Using default: standard")
            self.quality = "standard"
            
        if self.style not in self.VALID_STYLES:
            logger.warning(f"Invalid OPENAI_STYLE in environment: {self.style}. Using default: vivid")
            self.style = "vivid"
    
    def set_model(self, model: str) -> bool:
        """
        Set the model if valid, otherwise return False
        """
        if model in self.VALID_MODELS:
            self.model = model
            logger.info(f"Model set to: {model}")
            return True
        else:
            logger.warning(f"Invalid model: {model}. Valid options are: {', '.join(self.VALID_MODELS)}")
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
        model (str, optional): The model to use. Options: "dall-e-3" (default) or "dall-e-2"
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
        if model.lower() in ["dall-e-3", "dall-e-2"]:
            response = client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality if model.lower() == "dall-e-3" else None,
                style=style if model.lower() == "dall-e-3" else None,
                n=1,
                response_format="url"
            )
            
            # Extract the URL from the response
            image_url = response.data[0].url
            logger.info(f"Image generated successfully with OpenAI: {image_url}")
            return image_url
        else:
            logger.error(f"Unsupported OpenAI model: {model}")
            return None
            
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
        logger.info(f"Generating image with Gemini model, prompt: '{prompt}'")
        
        # Generate image with Gemini
        model = genai.GenerativeModel('gemini-2.0-flash-exp-image-generation')
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
    
    # Generate image based on the model
    if model.lower() == "gemini":
        return generate_image_with_gemini(prompt)
    elif model.lower() in ["dall-e-3", "dall-e-2"]:
        return generate_image_with_openai(prompt, model=model)
    else:
        logger.error(f"Unsupported model: {model}")
        return None


# Create a global instance of ImageGenConfig
gen_config = ImageGenConfig()
