# YAIL (Yet Another Image Loader)

## About ##
Atari 8bit image loader supporting binary PBM and its own YAI format.

If you have a FujiNet you can view streamed images from the search terms you enter.

Using custom display lists YAIL is able to display 220 lines instead of the default 192. This means that when loading a PBM (black and white) image the display will be in Graphics 8 (ANTIC F) at a 320x220 resolution.

## Console ##
YAIL has a simple text console for interaction that is activated when you start to type.
Commands:

  - help                  - List of commands
  - load <filename>       - Loads the specified PBM/PGM files and now a new YAI file.
  - save <filename>       - Saves the current image and graphics state to a YAI file.
  - cls                   - Clears the screen
  - gfx #  (0, 8, 9)      - Change the graphics mode to the number specified
  - stream <search terms> - Stream images (gfx 9) from the yailsrv.py.
  - set server <url>      - Give the N:TCP URL for the location of the yailsrv.py.
                             Ex: set server N:TCP://192.168.1.205:9999/
  - quit              - Quit the application

Tested on and works with the Atari 800XL.  Other models, **YMMV**

## Command line ##
Usage: YAIL.XEX [OPTIONS]

  -h this message
  
  -l <filename> load image file
  
  -u <url> use this server address
  
  -s <tokens> search terms

## Server ##
The server is written in Python and provides various image streaming capabilities for the YAIL client on Atari 8-bit computers via FujiNet.

### Features ###
- **OpenAI Image Generation**: Generate images using OpenAI's DALL-E 3 model based on text prompts
- **Local Image Streaming**: Stream images from a local directory
- **Web Camera Support**: Stream live video from a connected webcam
- **Multiple Graphics Modes**: Support for different Atari graphics modes (8, 9, and VBXE)
- **Custom Image Processing**: Automatically resize, crop, and format images for optimal display on Atari
- **HTTP Request Handling**: Properly responds to HTTP requests with appropriate messages
- **Network Detection**: Automatically detects available network interfaces and recommends the best IP for connections

### Requirements ###
- Python 3.6+
- Required Python packages (install via pip):
  - requests
  - duckduckgo_search
  - fastcore
  - pillow
  - tqdm
  - olefile
  - numpy
  - pygame
  - openai
  - python-dotenv
  - netifaces

### Server Commands ###
The YAIL server can process the following commands from clients:
- `generate <prompt>` or `gen <prompt>`: Generate an image using OpenAI's DALL-E 3 model
- `search <terms>`: Search for images using the provided terms
- `camera`: Stream from a connected webcam
- `openai`: Configure OpenAI settings
- `gfx <mode>`: Set the graphics mode
- `quit`: Exit the client connection

### Configuration ###
The server can be configured using environment variables or command-line arguments:

#### Environment Variables
Create a file named `env` in the server directory with the following variables:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=dall-e-3
OPENAI_SIZE=1024x1024
OPENAI_QUALITY=standard
OPENAI_STYLE=vivid
OPENAI_SYSTEM_PROMPT=You are an expert illustrator creating beautiful, imaginative artwork
```

#### Command-Line Arguments
```
--paths <directory>      Directory containing images to stream
--extensions <ext>       File extensions to include (default: .jpg, .jpeg, .gif, .png)
--camera <device>        Camera device to use
--port <number>          Port to listen on (default: 5556)
--loglevel <level>       Logging level (INFO, DEBUG, etc.)
--openai-api-key <key>   OpenAI API key
--openai-model <model>   OpenAI model (dall-e-3 or dall-e-2)
--openai-size <size>     Image size (1024x1024, 1792x1024, or 1024x1792)
--openai-quality <qual>  Image quality (standard or hd)
--openai-style <style>   Image style (vivid or natural)
```

### Deployment ###
The `deployment` directory contains scripts and configuration files to help deploy the YAIL server as a systemd service on Linux systems.

#### Deployment Files
- `fujinet-yail.service`: Systemd service file that properly activates the Python virtual environment
- `deploy.sh`: Installation script that sets up the service, environment, and dependencies
- `test_service.sh`: Script to test the YAIL server via curl, automatically detecting server IP and port
- `env.example`: Example environment configuration file

#### Deployment Instructions
1. Clone the repository
2. Navigate to the deployment directory
3. Run the deployment script:
   ```
   ./deploy.sh
   ```
4. The script will:
   - Create a Python virtual environment
   - Install required dependencies
   - Set up the systemd service
   - Configure environment variables

### Testing ###
The `deployment` directory also contains test scripts to verify the server's functionality:

#### Test Scripts
- `test_service.sh`: Tests basic connectivity to the YAIL server
- `test_gen_command.py`: Tests the image generation functionality
- `test_image_gen.py`: Advanced testing script with detailed binary data analysis
- `test_server_logs.py`: Monitors server logs during testing

#### Running Tests
```
# Test basic connectivity
./deployment/test_service.sh

# Test image generation
python deployment/test_gen_command.py "happy people dancing"
```

### Example Usage ###
1. Start the server with local images:
   ```
   python server/yail.py --paths /path/to/images --loglevel INFO
   ```

2. Start the server with OpenAI image generation:
   ```
   python server/yail.py --openai-api-key your_api_key_here --openai-model dall-e-3
   ```

3. Start the server as a systemd service:
   ```
   sudo systemctl start fujinet-yail
   ```
