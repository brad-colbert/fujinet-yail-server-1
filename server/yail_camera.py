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
YAIL_H = 192

# Global variables
camera_thread = None
camera_done = False
camera_name = None
camera_mutex = threading.Lock()
camera_image = None


def init_camera(device_name: Optional[str] = None) -> bool:
    """
    Initialize the camera with the specified device name.
    
    Args:
        device_name (str, optional): Camera device name. If None, uses the first available camera.
        
    Returns:
        bool: True if camera was initialized successfully, False otherwise
    """
    global camera_name
    
    if not PYGAME_AVAILABLE:
        logger.error("Cannot initialize camera: pygame not available")
        return False
    
    try:
        pygame.camera.init()
        camera_list = pygame.camera.list_cameras()
        
        if not camera_list:
            logger.error("No cameras found")
            return False
        
        # If device_name is provided, try to find it in the list
        if device_name:
            if device_name in camera_list:
                camera_name = device_name
                logger.info(f"Using specified camera: {camera_name}")
            else:
                logger.warning(f"Specified camera '{device_name}' not found. Available cameras: {camera_list}")
                camera_name = camera_list[0]
                logger.info(f"Using first available camera: {camera_name}")
        else:
            camera_name = camera_list[0]
            logger.info(f"Using first available camera: {camera_name}")
        
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
    global camera_name
    
    if not PYGAME_AVAILABLE:
        logger.error("Cannot capture image: pygame not available")
        return None
    
    if not camera_name:
        logger.error("Cannot capture image: camera not initialized")
        return None
    
    try:
        # Initialize the camera
        cam = pygame.camera.Camera(camera_name, (width, height))
        cam.start()
        
        # Allow the camera to warm up
        time.sleep(0.5)
        
        # Capture an image
        img = cam.get_image()
        
        # Convert pygame surface to PIL Image
        img_str = pygame.image.tostring(img, 'RGB')
        pil_img = Image.frombytes('RGB', img.get_size(), img_str)
        
        # Stop the camera
        cam.stop()
        
        return pil_img
    except Exception as e:
        logger.error(f"Error capturing image from camera: {e}")
        return None


def camera_handler(gfx_mode: int, process_image_func, update_data_func) -> None:
    """
    Camera handler thread function. Continuously captures images from the camera
    and processes them for streaming.
    
    Args:
        gfx_mode (int): Graphics mode to use for image processing
        process_image_func (callable): Function to process the captured image
        update_data_func (callable): Function to update the image data for streaming
    """
    global camera_done
    global camera_image
    
    if not PYGAME_AVAILABLE:
        logger.error("Cannot start camera handler: pygame not available")
        return
    
    logger.info(f"Starting camera handler thread with mode: {gfx_mode}")
    
    try:
        # Initialize pygame and the camera
        pygame.camera.init()
        
        if not camera_name:
            camera_list = pygame.camera.list_cameras()
            if not camera_list:
                logger.error("No cameras found")
                return
            camera_name = camera_list[0]
        
        logger.info(f"Using camera: {camera_name}")
        
        # Initialize the camera with the specified resolution
        cam = pygame.camera.Camera(camera_name, (YAIL_W, YAIL_H))
        cam.start()
        
        # Main camera loop
        while not camera_done:
            try:
                # Capture an image
                img = cam.get_image()
                
                # Convert pygame surface to PIL Image
                img_str = pygame.image.tostring(img, 'RGB')
                pil_img = Image.frombytes('RGB', img.get_size(), img_str)
                
                # Process the image based on the graphics mode
                processed_img = process_image_func(pil_img)
                
                # Update the global camera image
                with camera_mutex:
                    camera_image = processed_img
                
                # Update the image data for streaming
                update_data_func(processed_img, gfx_mode, thread_safe=True)
                
                # Sleep to control frame rate
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in camera loop: {e}")
                time.sleep(1)  # Sleep longer on error
        
        # Clean up
        cam.stop()
        pygame.camera.quit()
        logger.info("Camera handler thread stopped")
    
    except Exception as e:
        logger.error(f"Error in camera handler: {e}")
    
    logger.debug(f"camera_handler thread exiting {threading.get_native_id()}")


def start_camera_thread(gfx_mode: int, process_image_func, update_data_func) -> bool:
    """
    Start the camera handler thread.
    
    Args:
        gfx_mode (int): Graphics mode to use for image processing
        process_image_func (callable): Function to process the captured image
        update_data_func (callable): Function to update the image data for streaming
        
    Returns:
        bool: True if thread was started successfully, False otherwise
    """
    global camera_thread
    global camera_done
    
    if camera_thread is not None and camera_thread.is_alive():
        logger.warning("Camera thread is already running")
        return False
    
    # Reset the camera_done flag
    camera_done = False
    
    # Create and start the camera thread
    camera_thread = threading.Thread(
        target=camera_handler,
        args=(gfx_mode, process_image_func, update_data_func)
    )
    camera_thread.daemon = True
    camera_thread.start()
    
    logger.info("Camera thread started")
    return True


def stop_camera_thread() -> bool:
    """
    Stop the camera handler thread.
    
    Returns:
        bool: True if thread was stopped successfully, False otherwise
    """
    global camera_thread
    global camera_done
    
    if camera_thread is None or not camera_thread.is_alive():
        logger.warning("No camera thread is running")
        return False
    
    # Set the camera_done flag to signal the thread to stop
    camera_done = True
    
    # Wait for the thread to finish
    camera_thread.join(timeout=2.0)
    
    if camera_thread.is_alive():
        logger.warning("Camera thread did not stop in time")
        return False
    
    camera_thread = None
    logger.info("Camera thread stopped")
    return True


def get_camera_image() -> Optional[np.ndarray]:
    """
    Get the current camera image.
    
    Returns:
        numpy.ndarray: Current camera image or None if no image is available
    """
    global camera_image
    
    with camera_mutex:
        return camera_image
