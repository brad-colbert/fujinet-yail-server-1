#!/bin/bash

# Script to test the YAIL server using curl
# This script will detect the server's IP and port, then send test commands

# Set script to exit on error
set -e

# Default port
DEFAULT_PORT=5556
# Default timeout in seconds
TIMEOUT=5

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
  local color=$1
  local message=$2
  echo -e "${color}${message}${NC}"
}

# Function to detect the server's IP address
detect_server_ip() {
  print_message "$BLUE" "Detecting server IP address..."
  
  # Try to get the IP using socket method (most reliable)
  local ip=$(ip route get 8.8.8.8 2>/dev/null | awk '{print $7}' | head -1)
  
  # If that fails, try using hostname
  if [ -z "$ip" ]; then
    ip=$(hostname -I 2>/dev/null | awk '{print $1}')
  fi
  
  # If that fails too, try using ifconfig
  if [ -z "$ip" ]; then
    ip=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -1)
  fi
  
  # If all methods fail, use localhost
  if [ -z "$ip" ]; then
    print_message "$YELLOW" "Could not detect server IP. Using localhost (127.0.0.1)."
    ip="127.0.0.1"
  else
    print_message "$GREEN" "Detected server IP: $ip"
  fi
  
  echo "$ip"
}

# Function to check if the server is running on a specific IP and port
check_server() {
  local ip=$1
  local port=$2
  
  print_message "$BLUE" "Testing connection to YAIL server at $ip:$port..."
  
  # Try to connect to the server
  if nc -z -w$TIMEOUT $ip $port 2>/dev/null; then
    print_message "$GREEN" "Server is running at $ip:$port"
    return 0
  else
    print_message "$YELLOW" "Could not connect to server at $ip:$port"
    return 1
  fi
}

# Function to send a test command to the server
test_command() {
  local ip=$1
  local port=$2
  local command=$3
  local description=$4
  
  print_message "$BLUE" "Testing command: $description"
  
  # Send the command to the server using netcat
  # Note: We use a timeout to avoid hanging if the server doesn't respond
  result=$(echo -e "$command" | nc -w$TIMEOUT $ip $port 2>/dev/null)
  
  if [ $? -eq 0 ]; then
    print_message "$GREEN" "Command successful!"
    echo "$result" | head -5
    return 0
  else
    print_message "$RED" "Command failed!"
    return 1
  fi
}

# Main execution
print_message "$BLUE" "=== YAIL Server Test Script ==="

# Get server IP
SERVER_IP=$(detect_server_ip)
SERVER_PORT=$DEFAULT_PORT

# Check if port was provided as an argument
if [ $# -ge 1 ]; then
  SERVER_PORT=$1
fi

# Check if the server is running
if check_server $SERVER_IP $SERVER_PORT; then
  # Test basic commands
  test_command $SERVER_IP $SERVER_PORT "openai-config model" "Get current OpenAI model"
  test_command $SERVER_IP $SERVER_PORT "help" "Get help information"
  
  print_message "$GREEN" "Tests completed. The YAIL server appears to be functioning correctly."
else
  print_message "$RED" "Could not connect to the YAIL server. Please make sure it's running."
  exit 1
fi

exit 0
