import logging
import time
import whisper
import pyaudio
import numpy as np
import torch
import webrtcvad
import asyncio
import os
from config_manager import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self, bot, config_path: str = "audio_config.json"):
        """Initialize the audio transcriber."""
        self.bot = bot
        self.p = pyaudio.PyAudio()
        
        # Initialize configuration using ConfigManager
        self.config_manager = ConfigManager(config_path)
        
        # Define default configuration
        default_config = {
            "device": {
                "audio_device_index": 0,
                "sample_rate": 16000,
                "chunk_size": 480,
                "channels": 1
            },
            "whisper": {
                "model_name": "medium"
            },
            "voice_detection": {
                "vad_mode": 2,
                "silence_threshold": 20
            },
            "filtering": {
                "filtered_phrases": []
            }
        }
        
        # Helper function to get config value
        def get_config(section, key, default):
            # First try the new nested structure
            value = self.config_manager.get(f"audio_transcriber", {})
            if value and isinstance(value, dict):
                section_data = value.get(section, {})
                if isinstance(section_data, dict):
                    if key in section_data:
                        return section_data[key]
            
            # Fallback to flat structure
            flat_key = f"{key}"
            value = self.config_manager.get(flat_key, default)
            return value
        
        # Initialize audio configuration
        self.audio_config = {}
        
        # Device settings
        self.audio_config['audio_device_index'] = get_config('device', 'audio_device_index', 
            default_config['device']['audio_device_index'])
        self.audio_config['sample_rate'] = get_config('device', 'sample_rate', 
            default_config['device']['sample_rate'])
        self.audio_config['chunk_size'] = get_config('device', 'chunk_size', 
            default_config['device']['chunk_size'])
        self.audio_config['channels'] = get_config('device', 'channels', 
            default_config['device']['channels'])
        
        # Whisper settings
        self.audio_config['model_name'] = get_config('whisper', 'model_name', 
            default_config['whisper']['model_name'])
        
        # Voice detection settings
        self.audio_config['vad_mode'] = get_config('voice_detection', 'vad_mode', 
            default_config['voice_detection']['vad_mode'])
        self.audio_config['silence_threshold'] = get_config('voice_detection', 'silence_threshold', 
            default_config['voice_detection']['silence_threshold'])
        
        # Filtering settings
        self.audio_config['filtered_phrases'] = get_config('filtering', 'filtered_phrases', 
            default_config['filtering']['filtered_phrases'])

        logger.info(f"Initialized with config: {self.audio_config}")
        
        try:
            # Initialize audio stream with explicit type conversion
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=int(self.audio_config['channels']),  # Ensure integer
                rate=int(self.audio_config['sample_rate']),  # Ensure integer
                input=True,
                input_device_index=int(self.audio_config['audio_device_index']),  # Ensure integer
                frames_per_buffer=int(self.audio_config['chunk_size'])  # Ensure integer
            )
        except Exception as e:
            logger.error(f"Error initializing audio stream: {e}")
            logger.error(f"Audio config used: {self.audio_config}")
            raise
        
        self.vad = webrtcvad.Vad(int(self.audio_config['vad_mode']))  # Ensure integer
        self.buffer = []
        self.is_recording = True
        self.is_speaking = False
        self.silence_frames = 0
        self.SILENCE_THRESHOLD = int(self.audio_config['silence_threshold'])  # Ensure integer
        
        logger.info("Loading Whisper model...")
        if torch.cuda.is_available():
            try:
                logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")
                
                # GPU optimizations
                torch.cuda.empty_cache()
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                self.model = whisper.load_model(
                    self.audio_config['model_name'],
                    device="cuda",
                    download_root=os.path.expanduser("~/.cache/whisper")
                )
                
            except Exception as e:
                logger.error(f"Error configuring GPU: {e}")
                logger.info("Falling back to CPU...")
                self.model = whisper.load_model(
                    self.audio_config['model_name'],
                    device="cpu",
                    download_root=os.path.expanduser("~/.cache/whisper")
                )
        else:
            self.model = whisper.load_model(
                self.audio_config['model_name'],
                device="cpu",
                download_root=os.path.expanduser("~/.cache/whisper")
            )

    def filter_transcription(self, transcription: str) -> str:
        """Filter out unwanted phrases from transcription."""
        if not transcription:
            return transcription
                
        filtered_phrases = self.audio_config['filtered_phrases']
        filtered_text = transcription
            
        for phrase in filtered_phrases:
            if phrase in filtered_text:
                filtered_text = filtered_text.replace(phrase, "")
                
        # Normalize spaces after filtering
        return ' '.join(filtered_text.split())

    async def process_audio_chunk(self, audio_data: bytes):
        """
        Processes a chunk of audio by calling Whisper in a separate thread,
        then sends the resulting transcription to the bot if not empty.
        """
        try:
            # Convert raw audio data to float32
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Only process if it's at least half a second of audio
            if len(audio_np) > self.audio_config['sample_rate'] * 0.5:
                logger.debug("Processing audio chunk...")
                
                # Free up some GPU memory if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Whisper transcription options
                options = {
                    "language": "en",
                    "task": "transcribe",
                    "beam_size": 5,
                    "best_of": 5,
                    "fp16": torch.cuda.is_available(),
                    "temperature": [0.0, 0.2, 0.4, 0.6],
                    "compression_ratio_threshold": 2.4,
                    "condition_on_previous_text": True,
                    "initial_prompt": None
                }
                
                # Offload the transcribe call to a thread
                result = await asyncio.to_thread(self.model.transcribe, audio_np, **options)
                
                transcription = result["text"].strip()
                if transcription:
                    # Apply filtering before sending
                    filtered_transcription = self.filter_transcription(transcription)
                    if filtered_transcription:
                        await self.bot.send_transcription(filtered_transcription)
                
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            if "CUDA out of memory" in str(e) and torch.cuda.is_available():
                logger.info("Attempting to recover from CUDA memory error...")
                torch.cuda.empty_cache()
                await asyncio.sleep(1)

    def test_audio_levels(self, duration=3) -> bool:
        """Test audio levels for the selected device (blocking)."""
        logger.info(f"Testing audio levels for {duration} seconds...")
        start_time = time.time()
        max_volume = 0
        min_volume = float('inf')
        
        while time.time() - start_time < duration:
            data = self.stream.read(self.audio_config['chunk_size'], exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()
            max_volume = max(max_volume, volume)
            min_volume = min(min_volume, volume)
        
        logger.info(f"Audio test complete. Volume range: {min_volume:.0f} - {max_volume:.0f}")
        return bool(min_volume > 0 and max_volume < 32768)

    async def record_and_transcribe(self):
        """
        Continuously read audio data, run VAD, buffer speech segments, and
        submit complete speech segments to Whisper for transcription.
        """
        logger.info("Starting audio recording and transcription...")
        
        if not self.test_audio_levels():
            logger.warning("Warning: Audio levels may not be optimal. Please check your microphone.")
        
        while self.is_recording:
            try:
                # Read from the stream in a background thread
                data = await asyncio.to_thread(
                    self.stream.read,
                    self.audio_config['chunk_size'],
                    False  # exception_on_overflow
                )
                
                # VAD can also be done in a thread if needed
                is_speech = await asyncio.to_thread(
                    self.vad.is_speech, data, self.audio_config['sample_rate']
                )
                
                if is_speech:
                    if not self.is_speaking:
                        logger.debug("Speech detected")
                        self.is_speaking = True
                    self.buffer.append(data)
                    self.silence_frames = 0
                else:
                    if self.is_speaking:
                        self.silence_frames += 1
                        self.buffer.append(data)  # Keep some silence for natural speech
                        
                        if self.silence_frames >= self.SILENCE_THRESHOLD:
                            logger.debug("Speech segment complete")
                            if self.buffer:
                                # Process the collected audio
                                audio_segment = b''.join(self.buffer)
                                await self.process_audio_chunk(audio_segment)
                            # Reset for next utterance
                            self.buffer = []
                            self.is_speaking = False
                            self.silence_frames = 0

            except Exception as e:
                logger.error(f"Error in recording loop: {e}")
                await asyncio.sleep(1)

            # Yield to the event loop briefly
            await asyncio.sleep(0)

    def stop(self):
        """Stop the audio transcription service."""
        logger.info("Stopping audio transcription...")
        self.is_recording = False
        if hasattr(self, 'stream') and self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p') and self.p is not None:
            self.p.terminate()
        logger.info("Audio transcription stopped.")