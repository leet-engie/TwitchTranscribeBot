from twitchio.ext import commands
import os
from dotenv import load_dotenv
import json
import asyncio
import time
import whisper
import pyaudio
import numpy as np
import torch
import signal
import sys
import webrtcvad
import collections

load_dotenv()

class AudioTranscriber:
    def __init__(self, bot, config):
        self.bot = bot
        self.config = config
        self.p = pyaudio.PyAudio()
        self.validate_audio_device()
        self.model_name = 'medium'
        
        # Load phrase filters from config
        self.filtered_phrases = self.config.get('filtered_phrases', [
            "Thanks for watching!",
            "Like and subscribe"
        ])
        
        # Audio configuration
        self.config['sample_rate'] = 16000
        self.config['chunk_size'] = 480
        
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.config['sample_rate'],
            input=True,
            input_device_index=self.config['audio_device_index'],
            frames_per_buffer=self.config['chunk_size']
        )
        
        self.vad = webrtcvad.Vad(2)
        self.buffer = []
        self.is_recording = True
        self.is_speaking = False
        self.silence_frames = 0
        self.SILENCE_THRESHOLD = 20
        
        print("Loading Whisper model...")
        if torch.cuda.is_available():
            try:
                print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
                print(f"CUDA version: {torch.version.cuda}")
                
                # RTX 4090 specific optimizations
                torch.cuda.empty_cache()
                torch.cuda.memory_summary(device=None, abbreviated=False)
                
                # Set memory allocator for RTX 4090 (16GB VRAM)
                torch.cuda.set_per_process_memory_fraction(0.8)
                torch.backends.cuda.max_split_size_mb = 1024
                
                # Enable TF32 for better performance on Ampere/Ada architecture
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                self.model = whisper.load_model(
                    self.model_name,
                    device="cuda",
                    download_root=os.path.expanduser("~/.cache/whisper"),
                )
                
                # Enable CUDA graphs for repeated inference
                torch.cuda.is_available = lambda: True
                torch._C._jit_set_profiling_executor(True)
                
                print("\nGPU Memory Usage:")
                print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB")
                print(f"Cached: {torch.cuda.memory_reserved(0)/1024**3:.2f}GB")
                print(f"Max allocated: {torch.cuda.max_memory_allocated(0)/1024**3:.2f}GB")
                
            except Exception as e:
                print(f"Error configuring GPU: {e}")
                print("Falling back to CPU...")
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

    def validate_audio_device(self):
        """Validate the selected audio device"""
        print("\nAvailable audio input devices:")
        device_found = False
        
        for i in range(self.p.get_device_count()):
            dev_info = self.p.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                print(f"Device {i}: {dev_info['name']}")
                if i == self.config['audio_device_index']:
                    device_found = True
                    print(f"Selected device {i}: {dev_info['name']}")
        
        if not device_found:
            raise ValueError(f"Audio device index {self.config['audio_device_index']} not found")

    def test_audio_levels(self, duration=3):
        """Test audio levels for the selected device"""
        print(f"\nTesting audio levels for {duration} seconds...")
        start_time = time.time()
        max_volume = 0
        min_volume = float('inf')
        
        while time.time() - start_time < duration:
            data = self.stream.read(self.config['chunk_size'], exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()
            max_volume = max(max_volume, volume)
            min_volume = min(min_volume, volume)
            print(f"\rVolume range: {min_volume:.0f} - {max_volume:.0f}", end='', flush=True)
        
        print(f"\nAudio test complete. Volume range: {min_volume:.0f} - {max_volume:.0f}")
        return min_volume > 0 and max_volume < 32768

    def should_filter_phrase(self, transcription):
        """Check if the transcription contains any filtered phrases"""
        for phrase in self.filtered_phrases:
            if phrase.lower() in transcription.lower():
                return True, phrase
        return False, None

    async def process_audio_chunk(self, audio_data):
        try:
            audio_data = np.frombuffer(audio_data, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            if len(audio_data) > self.config['sample_rate'] * 0.5:
                print("\nProcessing audio chunk...")
                
                if torch.cuda.is_available():
                    print(f"GPU Memory before processing: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
                    torch.cuda.empty_cache()
                    
                    options = {
                        "language": "en",
                        "task": "transcribe",
                        "beam_size": 5,
                        "best_of": 5,
                        "fp16": True,
                        "temperature": [0.0, 0.2, 0.4, 0.6],
                        "compression_ratio_threshold": 2.4,
                        "condition_on_previous_text": True,
                        "initial_prompt": None
                    }
                    
                    scaler = torch.cuda.amp.GradScaler()
                    with torch.cuda.amp.autocast(enabled=True):
                        with torch.inference_mode():
                            result = self.model.transcribe(audio_data, **options)
                    
                    print(f"GPU Memory after processing: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
                else:
                    with torch.inference_mode():
                        result = self.model.transcribe(audio_data, **options)
                
                transcription = result["text"].strip()
                print(f"\nTranscription: {transcription}")
                
                should_filter, matched_phrase = self.should_filter_phrase(transcription)
                
                if should_filter:
                    print(f"\nFiltered out phrase containing '{matched_phrase}'")
                elif transcription:
                    channel = self.bot.get_channel(os.getenv('CHANNEL'))
                    if channel:
                        await channel.send(f"{self.bot.bot_prefix}{transcription}")
                
        except Exception as e:
            print(f"\nError in transcription: {e}")
            if "CUDA out of memory" in str(e) and torch.cuda.is_available():
                print("Attempting to recover from CUDA memory error...")
                torch.cuda.empty_cache()
                await asyncio.sleep(1)

    async def record_and_transcribe(self):
        print("\nRecording and transcribing... (Press Ctrl+C to stop)")
        
        if not self.test_audio_levels():
            print("Warning: Audio levels may not be optimal. Please check your microphone.")
        
        print("\nStarting audio processing loop...")
        while self.is_recording:
            try:
                # Read audio chunk
                data = self.stream.read(self.config['chunk_size'], exception_on_overflow=False)
                
                try:
                    # Check if it's speech
                    is_speech = self.vad.is_speech(data, self.config['sample_rate'])
                except Exception as e:
                    print(f"VAD error: {e}")
                    continue
                
                if is_speech:
                    if not self.is_speaking:
                        print("\nSpeech detected")
                        self.is_speaking = True
                    self.buffer.append(data)
                    self.silence_frames = 0
                else:
                    if self.is_speaking:
                        self.silence_frames += 1
                        self.buffer.append(data)  # Keep some silence for natural speech
                        
                        if self.silence_frames >= self.SILENCE_THRESHOLD:
                            print("\nSpeech segment complete")
                            if self.buffer:
                                # Process the collected audio
                                audio_segment = b''.join(self.buffer)
                                await self.process_audio_chunk(audio_segment)
                            # Reset for next utterance
                            self.buffer = []
                            self.is_speaking = False
                            self.silence_frames = 0
                
            except Exception as e:
                print(f"\nError in recording loop: {e}")
                await asyncio.sleep(1)

    def stop(self):
        print("\nStopping audio transcription...")
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
        print("Audio transcription stopped.")

class Bot(commands.Bot):
    def __init__(self):
        self.bot_prefix = "[🎙️]: "

        try:
            with open('audio_config.json', 'r') as f:
                self.audio_config = json.load(f)
        except FileNotFoundError:
            print("audio_config.json not found, using default values")
            self.audio_config = {
                "audio_device_index": 0,
                "sample_rate": 16000,
                "chunk_size": 480,
                "channels": 1,
                "filtered_phrases": [
                    "Thanks for watching!",
                    "Like and subscribe",
                    "Don't forget to subscribe"
                ]
            }
        
        super().__init__(
            token=os.getenv('TMI_TOKEN'),
            prefix='!',
            nick=os.getenv('BOT_NICK'),
            initial_channels=[os.getenv('CHANNEL')]
        )
        
        self.should_stop = False
        self.transcriber = None
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        print("\nShutdown signal received...")
        self.should_stop = True
        self.cleanup()
        sys.exit(0)

    async def event_ready(self):
        print(f'Logged in as | {self.nick}')
        print(f'Connected to channel: {os.getenv("CHANNEL")}')
        
        try:
            self.transcriber = AudioTranscriber(self, self.audio_config)
            self.transcribe_task = asyncio.create_task(self.transcriber.record_and_transcribe())
            
            channel = self.get_channel(os.getenv('CHANNEL'))
            await channel.send(f"{self.bot_prefix}Connected and listening!")
        except Exception as e:
            print(f"Error initializing transcriber: {e}")
            channel = self.get_channel(os.getenv('CHANNEL'))
            await channel.send(f"{self.bot_prefix}Failed to initialize audio transcription.")

    def cleanup(self):
        print("\nCleaning up resources...")
        if hasattr(self, 'transcribe_task'):
            self.transcribe_task.cancel()
        if hasattr(self, 'transcriber'):
            self.transcriber.stop()
        print("Cleanup complete.")

async def main():
    bot = Bot()
    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received...")
    except Exception as e:
        print(f"Error running bot: {e}")
    finally:
        if bot:
            bot.cleanup()
        print("Bot shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nForced exit requested...")
        sys.exit(0)