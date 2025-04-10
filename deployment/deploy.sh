#!/bin/bash

# Script to deploy the FujiNet YAIL Server systemd service
# This script should be run with sudo privileges

# Exit on error
set -e

# Configuration
SERVICE_NAME="fujinet-yail"
SERVICE_FILE="${SERVICE_NAME}.service"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_SOURCE="${SCRIPT_DIR}/${SERVICE_FILE}"
SERVICE_DEST="/etc/systemd/system/${SERVICE_FILE}"

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script with sudo privileges"
  exit 1
fi

# Function to check if service exists and if it's different from our version
check_service() {
  if [ -f "$SERVICE_DEST" ]; then
    echo "Service file already exists at $SERVICE_DEST"
    
    # Check if files are different
    if diff -q "$SERVICE_SOURCE" "$SERVICE_DEST" >/dev/null; then
      echo "Existing service file is identical to the new one. No changes needed."
      return 1  # No changes needed
    else
      echo "Existing service file is different from the new one. Will update."
      return 0  # Update needed
    fi
  else
    echo "Service file does not exist. Will install."
    return 0  # Install needed
  fi
}

# Function to install or update service
install_service() {
  echo "Installing/updating systemd service..."
  
  # Copy service file to systemd directory
  cp "$SERVICE_SOURCE" "$SERVICE_DEST"
  
  # Set proper permissions
  chmod 644 "$SERVICE_DEST"
  
  # Reload systemd to recognize the new service
  systemctl daemon-reload
  
  echo "Service file installed at $SERVICE_DEST"
}

# Function to restart service if it's already running
restart_service() {
  if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "Stopping existing $SERVICE_NAME service..."
    systemctl stop "$SERVICE_NAME"
  fi
  
  echo "Starting $SERVICE_NAME service..."
  systemctl start "$SERVICE_NAME"
  
  echo "Enabling $SERVICE_NAME service to start on boot..."
  systemctl enable "$SERVICE_NAME"
  
  echo "Service status:"
  systemctl status "$SERVICE_NAME"
}

# Main execution
echo "Deploying FujiNet YAIL Server systemd service..."

# Check if we need to update the service
if check_service; then
  install_service
  restart_service
else
  # Even if the service file is the same, make sure it's enabled and running
  if ! systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "Service is not running. Starting it..."
    systemctl start "$SERVICE_NAME"
    systemctl enable "$SERVICE_NAME"
  else
    echo "Service is already running and up to date."
  fi
fi

echo "Deployment completed successfully!"
