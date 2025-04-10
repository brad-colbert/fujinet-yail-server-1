#!/usr/bin/env python

import socket
import time
import sys
import os
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def send_command(host, port, command, timeout=30):
    """
    Send a command to the YAIL server and log the response.
    
    Args:
        host (str): Server hostname or IP
        port (int): Server port
        command (str): Command to send
        timeout (int): Socket timeout in seconds
        
    Returns:
        bool: True if successful, False otherwise
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    
    try:
        # Connect to the server
        sock.connect((host, port))
        logger.info(f"Connected to server at {host}:{port}")
        
        # Send the command
        logger.info(f"Sending command: {command}")
        sock.sendall(command.encode())
        
        # Wait for initial response
        time.sleep(1)
        
        # Send a newline to acknowledge the initial response
        sock.sendall(b'\n')
        
        # Wait for the server to process the command
        logger.info("Waiting for server to process the command...")
        time.sleep(5)
        
        # Check if the server is still responding
        sock.settimeout(5)
        try:
            data = sock.recv(1024)
            if data:
                logger.info(f"Server is still responding: {len(data)} bytes received")
                return True
        except socket.timeout:
            logger.info("No additional data received, which is expected")
        
        return True
        
    except ConnectionRefusedError:
        logger.error(f"Could not connect to {host}:{port}. Is the server running?")
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        return False
    finally:
        sock.close()
        logger.info("Connection closed")

def main():
    """
    Test the 'gen' command on the YAIL server.
    """
    # Server connection details
    host = '127.0.0.1'  # localhost
    port = 5556         # default YAIL server port
    
    # Default prompt
    prompt = "happy people dancing in a circle"
    if len(sys.argv) > 1:
        prompt = ' '.join(sys.argv[1:])
    
    # Test the 'gen' command
    command = f"gen {prompt}"
    success = send_command(host, port, command)
    
    if success:
        logger.info("Test completed successfully")
        logger.info("Check the server logs for details about image generation")
    else:
        logger.error("Test failed")

if __name__ == "__main__":
    main()
