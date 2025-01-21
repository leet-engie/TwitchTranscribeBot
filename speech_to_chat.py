import logging
import time
import whisper
import pyaudio
import numpy as np
import torch
import webrtcvad
from typing import Optional
import asyncio
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self, bot):
        """Initialize the audio transcriber."""
        self.bot = bot
        self.p = pyaudio.PyAudio()
        self.model_name = 'medium'
        
        # Load audio configuration
        try:
            with open('audio_config.json', 'r') as f:
                self.audio_config = json.load(f)
            logger.info(f"Loaded audio configuration from audio_config.json")
        except FileNotFoundError:
            logger.warning("audio_config.json not found, using default values")
            self.audio_config = {
                "audio_device_index": 0,
                "sample_rate": 16000,
                "chunk_size": 480,
                "channels": 1,
                "filtered_phrases": []
            }
        
        # Set fixed sample rate and chunk size for VAD
        self.audio_config['sample_rate'] = 16000
        self.audio_config['chunk_size'] = 480
        
        # Initialize audio stream
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.audio_config['sample_rate'],
            input=True,
            input_device_index=self.audio_config['audio_device_index'],
            frames_per_buffer=self.audio_config['chunk_size']
        )
        
        self.vad = webrtcvad.Vad(2)
        self.buffer = []
        self.is_recording = True
        self.is_speaking = False
        self.silence_frames = 0
        self.SILENCE_THRESHOLD = 20
        
        logger.info("Loading Whisper model...")
        if torch.cuda.is_available():
            try:
                logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")
                
                # GPU optimizations
                torch.cuda.empty_cache()
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                self.model = whisper.load_model(
                    self.model_name,
                    device="cuda",
                    download_root=os.path.expanduser("~/.cache/whisper")
                )
                
            except Exception as e:
                logger.error(f"Error configuring GPU: {e}")
                logger.info("Falling back to CPU...")
                self.model = whisper.load_model(
                    self.model_name,
                    device="cpu",
                    download_root=os.path.expanduser("~/.cache/whisper")
                )
        else:
            self.model = whisper.load_model(
                self.model_name,
                device="cpu",
                download_root=os.path.expanduser("~/.cache/whisper")
            )

    def test_audio_levels(self, duration=3):
        """Test audio levels for the selected device"""
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

    async def process_audio_chunk(self, audio_data):
        try:
            audio_data = np.frombuffer(audio_data, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            if len(audio_data) > self.audio_config['sample_rate'] * 0.5:
                logger.debug("Processing audio chunk...")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
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
                
                with torch.inference_mode():
                    result = self.model.transcribe(audio_data, **options)
                
                transcription = result["text"].strip()
                if transcription:
                    await self.bot.send_transcription(transcription)
                
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            if "CUDA out of memory" in str(e) and torch.cuda.is_available():
                logger.info("Attempting to recover from CUDA memory error...")
                torch.cuda.empty_cache()
                await asyncio.sleep(1)

    async def record_and_transcribe(self):
        logger.info("Starting audio recording and transcription...")
        
        if not self.test_audio_levels():
            logger.warning("Warning: Audio levels may not be optimal. Please check your microphone.")
        
        while self.is_recording:
            try:
                data = self.stream.read(self.audio_config['chunk_size'], exception_on_overflow=False)
                
                try:
                    is_speech = self.vad.is_speech(data, self.audio_config['sample_rate'])
                except Exception as e:
                    logger.error(f"VAD error: {e}")
                    continue
                
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

    def stop(self):
        """Stop the audio transcription service."""
        logger.info("Stopping audio transcription...")
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
        logger.info("Audio transcription stopped.")