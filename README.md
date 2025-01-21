# Speech to Twitch Chat Transcriber

A Python-based tool that transcribes speech from your microphone directly to Twitch chat using OpenAI's Whisper model.

## Features
- Real-time speech-to-text transcription using Whisper  
- Automatic voice activity detection (VAD)  
- GPU acceleration support for faster transcription  
- Configurable audio device selection  
- Phrase filtering to prevent unwanted messages  
- Twitch chat integration  

## Prerequisites
- Python 3.7+  
- CUDA-compatible GPU (optional, but recommended)  
- Working microphone  
- Twitch account and bot credentials  

## Environment Variables
Create a `.env` file with the following variables:

```
TMI_TOKEN=your_twitch_oauth_token
BOT_NICK=your_bot_username
CHANNEL=target_twitch_channel
```

## Installation

1. **Install required packages:**
    ```bash
    pip install twitchio pyaudio whisper torch webrtcvad numpy python-dotenv
    ```

2. **For NVIDIA GPU support:**
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

## Configuration
The `audio_config.json` file controls the audio settings and filtered phrases:

```json
{
    "audio_device_index": 0,
    "sample_rate": 16000,
    "chunk_size": 4096,
    "channels": 1,
    "filtered_phrases": [
        "Thanks for watching!",
        "Like and subscribe"
    ]
}
```

## Usage

1. **Run the audio device detection tool to select your microphone:**
    ```bash
    python detect_audio_device.py
    ```

2. **Start the transcription bot:**
    ```bash
    python speech_to_chat.py
    ```

## Audio Device Selection
The `detect_audio_device.py` script provides:
- List of all available audio input devices  
- Individual device testing  
- Automatic device quality assessment  
- Configuration saving  

## Features of the Transcription Bot
- Real-time voice activity detection  
- Automatic silence detection  
- GPU memory optimization for RTX cards  
- Configurable phrase filtering  
- Error recovery and graceful shutdown  
- Memory-efficient audio processing  

## Troubleshooting
- If no audio devices are detected, check your microphone connections  
- For CUDA out-of-memory errors, try reducing the model size from `medium` to `base`  
- Ensure your Twitch bot has proper channel permissions  

## Notes
- The bot prefixes all transcribed messages with "🎙️"  
- Default sample rate is 16kHz for optimal Whisper performance  
- GPU acceleration significantly improves transcription speed  
- The bot automatically recovers from audio and CUDA errors  
```
