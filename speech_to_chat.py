import logging
import time
import whisper
import pyaudio
import numpy as np
import torch
import webrtcvad
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
        
        # Fix sample rate and chunk size for VAD
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

    def filter_transcription(self, transcription: str) -> str:
        """Filter out unwanted phrases from transcription."""
        if not transcription:
            return transcription
            
        filtered_phrases = self.audio_config.get("filtered_phrases", [])
        filtered_text = transcription
        
        for phrase in filtered_phrases:
            if phrase in filtered_text:
                filtered_text = filtered_text.replace(phrase, "").strip()
                logger.debug(f"Filtered out phrase: {phrase}")
                
        return filtered_text

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
                
                # Whisper transcription is also blocking -> do it in a thread
                options = {
                    "language": "en",  # Adjust if not always English
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
                    if filtered_transcription:  # Only send if there's text after filtering
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
        
        By using asyncio.to_thread(...) for the blocking parts, we ensure
        the main event loop can still process chat commands in parallel.
        """
        logger.info("Starting audio recording and transcription...")
        
        # It's okay to do a short blocking test before we enter the loop
        # but if you want it fully async, wrap this too
        if not self.test_audio_levels():
            logger.warning("Warning: Audio levels may not be optimal. Please check your microphone.")
        
        while self.is_recording:
            try:
                # 1) Read from the stream in a background thread
                data = await asyncio.to_thread(
                    self.stream.read,
                    self.audio_config['chunk_size'],
                    False  # exception_on_overflow
                )
                
                # 2) VAD can also be done in a thread if needed
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
                # Sleep a bit before trying to read again
                await asyncio.sleep(1)

            # Yield to the event loop briefly so other tasks can run
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
