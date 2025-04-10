#!/usr/bin/env python

import socket
import time
import sys

def connect_to_server(host='127.0.0.1', port=5556):
    """Connect to the YAIL server and return the socket."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))
        print(f"Connected to server at {host}:{port}")
        return s
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        sys.exit(1)

def send_command(sock, command):
    """Send a command to the server and print the response."""
    print(f"Sending command: {command}")
    sock.sendall(command.encode('utf-8'))
    time.sleep(1)  # Give the server time to process
    response = sock.recv(1024)
    print(f"Response: {response.decode('utf-8', errors='ignore')}")
    return response

def main():
    # Connect to the server
    sock = connect_to_server()
    
    # Generate an image with the prompt specified
    prompt = "happy people dancing in a circle"
    send_command(sock, f"gen {prompt}")
    
    # Wait for the server to process
    print("Waiting for image generation...")
    time.sleep(20)  # Increased wait time to allow for image generation
    
    # Try to receive any additional data
    try:
        sock.settimeout(5)
        while True:
            data = sock.recv(1024)
            if not data:
                break
            print(f"Additional response: {data.decode('utf-8', errors='ignore')}")
    except socket.timeout:
        print("No more data received")
    
    # Close the connection
    sock.close()
    print("Connection closed")

if __name__ == "__main__":
    main()
