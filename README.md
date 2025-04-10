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
- **OpenAI Image Generation**: Generate images using OpenAI's DALL-E 3 or GPT-4o models based on text prompts
- **Local Image Streaming**: Stream images from a local directory
- **Web Camera Support**: Stream live video from a connected webcam
- **Multiple Graphics Modes**: Support for different Atari graphics modes (8, 9, and VBXE)
- **Custom Image Processing**: Automatically resize, crop, and format images for optimal display on Atari

### Requirements ###
- Python 3.6+
- Required Python packages (install via pip):
  - requests
  - duckduckgo_search
  - fastcore
  - pillow
  - tqdm
  - numpy
  - openai
  - python-dotenv
  - pygame (for certain features)

### Installation ###
1. Clone this repository:
   ```
   git clone https://github.com/dillera/fujinet-yail-server.git
   cd fujinet-yail-server
   ```

2. Install required Python packages:
   ```
   cd server
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   python deployment/create_env.py
   ```
   This will create a `env` file with your OpenAI API key and other configuration options.

### Starting the Server ###
Basic usage:
```
python yail.py
```

With command line options:
```
python yail.py [--paths DIRECTORY] [--extensions jpg jpeg png gif] [--camera DEVICE] [--port PORT] [--loglevel LEVEL] [--openai-api-key KEY] [--openai-model MODEL] [--openai-size SIZE] [--openai-quality QUALITY] [--openai-style STYLE]
```

#### Command Line Arguments ####
- `--paths`: Directory path or list of file paths to serve images from
- `--extensions`: List of file extensions to process (default: .jpg, .jpeg, .gif, .png)
- `--camera`: Camera device to use for video streaming
- `--port`: Port to listen on (default: 5556)
- `--loglevel`: Logging level (DEBUG, INFO, WARN, ERROR, CRITICAL)
- `--openai-api-key`: OpenAI API key for image generation
- `--openai-model`: OpenAI model to use (dall-e-3 or gpt-4o)
- `--openai-size`: Image size for DALL-E 3 (1024x1024, 1792x1024, or 1024x1792)
- `--openai-quality`: Image quality for DALL-E 3 (standard or hd)
- `--openai-style`: Image style for DALL-E 3 (vivid or natural)

### Client Commands ###
The YAIL server accepts the following commands from clients:

- `generate <prompt>`: Generate an image using OpenAI based on the text prompt
- `search <terms>`: Legacy command, now redirects to OpenAI image generation
- `files`: Stream a random image from the loaded local files
- `video`: Stream video from a connected webcam
- `next`: Get the next image based on the current mode (generate, files, video)
- `gfx <mode>`: Set the graphics mode (8, 9, or 17 for VBXE)
- `openai-config [param] [value]`: Configure OpenAI settings:
  - `model`: Set model (dall-e-3 or gpt-4o)
  - `size`: Set image size (1024x1024, 1792x1024, or 1024x1792)
  - `quality`: Set quality (standard or hd)
  - `style`: Set style (vivid or natural)
  - `system_prompt`: Set system prompt for image generation
- `quit`: Close the connection

### Environment Variables ###
You can configure the server using environment variables in the `env` file:

```
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: OpenAI Model Configuration
OPENAI_MODEL=dall-e-3
OPENAI_SIZE=1024x1024
OPENAI_QUALITY=standard
OPENAI_STYLE=vivid
OPENAI_SYSTEM_PROMPT="You are an image generation assistant. Generate an image based on the user's description."
```

### Example Usage ###
1. Start the server with local images:
   ```
   python yail.py --paths /path/to/images --loglevel INFO
   ```

2. Start the server with OpenAI image generation:
   ```
   python yail.py --openai-api-key your_api_key --openai-model dall-e-3 --loglevel INFO
   ```

3. Connect from an Atari 8-bit computer with FujiNet and YAIL.XEX:
   - Set the server URL in YAIL: `set server N:TCP://your_server_ip:5556/`
   - Generate an image: `stream a colorful sunset over mountains`
   - Request next image: `next`

### Test Client ###
A simple test client (`testclient.py`) is included for testing the server without an Atari:
```python
# Modify the IP address and port as needed
client.connect(('192.168.1.126', 5556))
# Change the command as needed (e.g., 'generate mountains', 'files', etc.)
client.send(b'search funny\n')
```

### Troubleshooting ###
- If images aren't displaying correctly, try changing the graphics mode with `gfx 8` or `gfx 9`
- For OpenAI API issues, check your API key and quota
- Use `--loglevel DEBUG` for detailed server logs to diagnose issues
