#!/usr/bin/env python3
"""
YAIL Camera Module

This module contains the camera functionality for the YAIL server,
including camera initialization, image capture, and processing.
"""

import os
import time
import logging
import threading
from typing import Optional, Tuple
import numpy as np
from PIL import Image

# Set up logging
logger = logging.getLogger(__name__)

# Try to import pygame for camera support
try:
    import pygame
    import pygame.camera
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logger.warning("Pygame library not available. Install with: pip install pygame")

# Constants
SOCKET_WAIT_TIME = 1
YAIL_W = 320
YAIL_H = 220

# Global variables
#camera_thread = None
#camera_done = False
#camera_name = None
#camera_mutex = threading.Lock()
#camera_image = None

cam = None


def init_camera(device_name: Optional[str] = None) -> bool:
    """
    Initialize the camera with the specified device name.
    
    Args:
        device_name (str, optional): Camera device name. If None, uses the first available camera.
        
    Returns:
        bool: True if camera was initialized successfully, False otherwise
    """
    #global camera_name
    global cam
    
    if not PYGAME_AVAILABLE:
        logger.error("Cannot initialize camera: pygame not available")
        return False
    
    try:
        # Initialize pygame and the camera
        pygame.camera.init()
        
        if not device_name:
            camera_list = pygame.camera.list_cameras()
            if not camera_list:
                logger.error("No cameras found")
                return False
            device_name = camera_list[0]  # just use the first camera
        
        logger.info(f"Using camera: {device_name}")
        
        # Initialize the camera with the specified resolution
        cam = pygame.camera.Camera(device_name, (YAIL_W, YAIL_H))
        cam.start()  # start the camera

        return True

    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return False


def capture_camera_image(width: int = YAIL_W, height: int = YAIL_H) -> Optional[Image.Image]:
    """
    Capture an image from the camera.
    
    Args:
        width (int): Width of the captured image
        height (int): Height of the captured image
        
    Returns:
        PIL.Image.Image: Captured image or None if capture failed
    """
    #global camera_name
    global cam
    
    if not PYGAME_AVAILABLE:
        logger.error("Cannot capture image: pygame not available")
        return None
    
    if not cam:
        logger.error("Cannot capture image: camera not initialized")
        return None
    
    try:
        # Capture an image
        img = cam.get_image()
        
        # Convert pygame surface to PIL Image
        img_str = pygame.image.tostring(img, 'RGB')
        pil_img = Image.frombytes('RGB', img.get_size(), img_str)

        return pil_img
    
    except Exception as e:
        logger.error(f"Error capturing image from camera: {e}")
        return None
    
def shutdown_camera() -> None:
    """
    Shutdown the camera and release resources.
    """
    global cam
    
    if not PYGAME_AVAILABLE:
        logger.error("Cannot shutdown camera: pygame not available")
        return
    
    try:
        if cam:
            cam.stop()
            cam = None
        pygame.camera.quit()
        logger.info("Camera shutdown successfully")
    except Exception as e:
        logger.error(f"Error shutting down camera: {e}")