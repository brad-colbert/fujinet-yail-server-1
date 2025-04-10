#!/usr/bin/env python

import socket
import time
import sys
import os
import re
import binascii

def main():
    """
    Test script for the YAIL server image generation functionality.
    Connects to the server, sends a 'gen' command with a prompt, and displays the response.
    """
    # Server connection details
    host = '127.0.0.1'  # localhost
    port = 5556         # default YAIL server port
    
    # Create a TCP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Connect to the server
        sock.connect((host, port))
        print(f"Connected to server at {host}:{port}")
        
        # Prepare the command
        prompt = "happy people dancing in a circle"
        if len(sys.argv) > 1:
            prompt = ' '.join(sys.argv[1:])
        
        command = f"gen {prompt}"
        print(f"Sending command: {command}")
        
        # Send the command
        sock.sendall(command.encode())
        
        # Receive and print the response
        print("Waiting for server response...")
        
        # Set a timeout for receiving data
        sock.settimeout(30)  # 30 seconds timeout
        
        # Buffer to collect all data
        all_data = bytearray()
        
        # Continuously receive data until timeout or connection closed
        while True:
            try:
                data = sock.recv(4096)
                if not data:
                    break
                
                all_data.extend(data)
                print(f"Received {len(data)} bytes, total: {len(all_data)} bytes")
                
                # Try to extract any URLs from the data
                url_pattern = re.compile(b'https?://[^\s]+')
                urls = url_pattern.findall(data)
                for url in urls:
                    try:
                        decoded_url = url.decode('utf-8', errors='ignore')
                        print(f"Image URL detected: {decoded_url}")
                    except Exception as e:
                        print(f"Error decoding URL: {e}")
                
                # Try to extract any text messages
                try:
                    # Look for common text patterns in the binary data
                    text_chunks = re.findall(b'[A-Za-z0-9 .,!?;:\'"-]{5,}', data)
                    for chunk in text_chunks:
                        decoded = chunk.decode('utf-8', errors='ignore')
                        if any(keyword in decoded.lower() for keyword in ['error', 'fail', 'success', 'image', 'generate']):
                            print(f"Message: {decoded}")
                except Exception as e:
                    print(f"Error extracting text: {e}")
                    
            except socket.timeout:
                print("Timeout waiting for more data")
                break
        
        # Print a hex dump of the first 200 bytes for debugging
        if all_data:
            print("\nHex dump of first 200 bytes:")
            hex_dump = binascii.hexlify(all_data[:200]).decode('ascii')
            for i in range(0, len(hex_dump), 32):
                print(hex_dump[i:i+32])
            
    except ConnectionRefusedError:
        print(f"Error: Could not connect to {host}:{port}. Is the server running?")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()
        print("Connection closed")

if __name__ == "__main__":
    main()
