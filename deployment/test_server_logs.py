#!/usr/bin/env python

import subprocess
import time
import socket
import sys
import os
import signal
import threading

def start_server():
    """Start the YAIL server in a subprocess and return the process"""
    server_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'server')
    env_path = os.path.join(server_dir, 'env')
    
    # Set up environment variables
    env = os.environ.copy()
    if os.path.exists(env_path):
        print(f"Using environment file: {env_path}")
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env[key] = value
    
    # Start the server process
    server_script = os.path.join(server_dir, 'yail.py')
    process = subprocess.Popen(
        [sys.executable, server_script],
        cwd=server_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=env
    )
    
    print(f"Started YAIL server (PID: {process.pid})")
    return process

def log_server_output(process):
    """Log the server output in a separate thread"""
    def reader():
        for line in process.stdout:
            print(f"SERVER: {line.strip()}")
    
    thread = threading.Thread(target=reader)
    thread.daemon = True
    thread.start()
    return thread

def send_command(command, host='127.0.0.1', port=5556, timeout=5):
    """Send a command to the YAIL server"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    
    try:
        # Connect to the server
        sock.connect((host, port))
        print(f"Connected to server at {host}:{port}")
        
        # Send the command
        print(f"Sending command: {command}")
        sock.sendall(command.encode())
        
        # Wait for response
        time.sleep(1)
        
        # Send a newline to acknowledge
        sock.sendall(b'\n')
        
        # Wait for the server to process
        print("Waiting for server to process the command...")
        time.sleep(5)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()
        print("Connection closed")

def main():
    """Main function to test the YAIL server"""
    try:
        # Start the server
        server_process = start_server()
        
        # Log server output
        log_thread = log_server_output(server_process)
        
        # Wait for server to initialize
        print("Waiting for server to initialize...")
        time.sleep(5)
        
        # Default prompt
        prompt = "happy people dancing in a circle"
        if len(sys.argv) > 1:
            prompt = ' '.join(sys.argv[1:])
        
        # Test the 'gen' command
        command = f"gen {prompt}"
        send_command(command)
        
        # Keep the script running to see server logs
        print("\nServer is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        if 'server_process' in locals():
            server_process.terminate()
            server_process.wait()
            print("Server stopped")

if __name__ == "__main__":
    main()
