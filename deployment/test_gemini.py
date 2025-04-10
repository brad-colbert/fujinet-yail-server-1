#!/usr/bin/env python3
"""
Test script for Google Gemini image generation.
This script tests the Gemini image generation API directly.
"""

import os
import sys
import time
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from the server's env file
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'server', 'env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")
else:
    print(f"No env file found at {env_path}. Using default environment variables.")

# Check for Gemini API key
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    print("Please set the GEMINI_API_KEY environment variable.")
    sys.exit(1)

try:
    import google.generativeai as genai
    print("Google Generative AI library is available")
except ImportError:
    print("Error: Google Generative AI library not available.")
    print("Install with: pip install google-generativeai")
    sys.exit(1)

def generate_image_with_gemini(prompt: str) -> str:
    """
    Generate an image using Google's Gemini API and save it to a file.
    
    Args:
        prompt (str): The text prompt to generate an image from
        
    Returns:
        str: Path to the saved image or None if generation failed
    """
    try:
        print(f"Generating image with Gemini model, prompt: '{prompt}'")
        
        # Configure the Gemini API
        genai.configure(api_key=gemini_api_key)
        
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
                    os.makedirs('test_images', exist_ok=True)
                    
                    # Generate a unique filename based on timestamp
                    timestamp = int(time.time())
                    image_path = f"test_images/gemini-{timestamp}.png"
                    
                    # Save the image
                    image.save(image_path)
                    image_saved = True
                    
                    print(f"Image generated successfully with Gemini: {image_path}")
                    
                    # Display the image if possible
                    try:
                        image.show()
                    except Exception as e:
                        print(f"Could not display image: {e}")
                    
                    break
                elif hasattr(part, 'text') and part.text:
                    print(f"Gemini text response: {part.text}")
        else:
            print("No candidates in Gemini response")
        
        if image_saved and image_path:
            return os.path.abspath(image_path)
        else:
            print("Failed to extract image from Gemini response")
            return None
            
    except Exception as e:
        print(f"Error generating image with Gemini: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_gemini.py \"your prompt here\"")
        sys.exit(1)
    
    prompt = sys.argv[1]
    image_path = generate_image_with_gemini(prompt)
    
    if image_path:
        print(f"Image generated and saved to: {image_path}")
    else:
        print("Failed to generate image")
        sys.exit(1)
