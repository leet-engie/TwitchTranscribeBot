import pyaudio
import whisper
import numpy as np
import json
import time
from pathlib import Path

class AudioDeviceTester:
    def __init__(self):
        # Suppress ALSA errors
        import os
        import sys
        import ctypes
        
        # Disable ALSA error output
        ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                            ctypes.c_char_p, ctypes.c_int,
                                            ctypes.c_char_p)

        def py_error_handler(filename, line, function, err, fmt):
            pass

        c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
        try:
            asound = ctypes.cdll.LoadLibrary('libasound.so.2')
            asound.snd_lib_error_set_handler(c_error_handler)
        except:
            pass  # If ALSA is not installed, just continue
        
        self.p = pyaudio.PyAudio()
        self.model = whisper.load_model("base")
        self.config_path = Path("audio_config.json")
        self.sample_rate = 16000
        self.chunk_size = 1024 * 4
        self.channels = 1
        self.test_duration = 5  # seconds
        
    def display_menu(self):
        """Display the main menu options."""
        print("\n" + "="*50)
        print("AUDIO DEVICE SELECTION MENU")
        print("="*50)
        print("1. Display all audio input devices")
        print("2. Test a specific device")
        print("3. Run optimal device test (tests all devices)")
        print("4. Exit")
        print("="*50)

    def list_all_devices(self):
        """List all compatible audio input devices in a pretty format."""
        print("\n📱 Scanning audio input devices...")
        print("\n🎤 Compatible Audio Input Devices:")
        print("=" * 60)
        
        input_devices = []
        found_devices = False
        
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                if self.validate_device(device_info):
                    found_devices = True
                    input_devices.append(device_info)
                    print(f"📍 Device [{i}]: {device_info['name']}")
                    print(f"   ├── Channels: {device_info['maxInputChannels']}")
                    print(f"   └── Sample Rate: {int(device_info['defaultSampleRate'])} Hz")
                    print("-" * 60)
        
        if not found_devices:
            print("❌ No compatible audio input devices found!")
            print("=" * 60)
            
        return input_devices

    def validate_device(self, device_info):
        """Test if a device can be opened with our required configuration."""
        import os
        import sys
        import tempfile

        # Redirect stderr to devnull temporarily
        stderr = sys.stderr
        devnull = open(os.devnull, 'w')
        sys.stderr = devnull

        try:
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_info['index'],
                frames_per_buffer=self.chunk_size
            )
            stream.stop_stream()
            stream.close()
            result = True
        except Exception:
            result = False
        
        # Restore stderr
        sys.stderr = stderr
        devnull.close()
        
        return result

    def calculate_audio_metrics(self, audio_data):
        """Calculate various audio quality metrics."""
        metrics = {}
        
        # Calculate signal level (volume)
        metrics['signal_level'] = np.abs(audio_data).mean()
        
        # Calculate signal-to-noise ratio (SNR)
        sorted_amplitudes = np.sort(np.abs(audio_data))
        noise_estimate = np.mean(sorted_amplitudes[:len(sorted_amplitudes)//10])
        signal_estimate = np.mean(sorted_amplitudes[-len(sorted_amplitudes)//10:])
        metrics['snr'] = 20 * np.log10(signal_estimate / noise_estimate) if noise_estimate > 0 else 0
        
        # Calculate dynamic range
        metrics['dynamic_range'] = np.ptp(audio_data)
        
        # Calculate zero crossings (rough frequency estimate)
        zero_crossings = np.sum(np.diff(np.signbit(audio_data).astype(int)) != 0)
        metrics['zero_crossing_rate'] = zero_crossings / len(audio_data)
        
        return metrics
        
    def score_device(self, metrics, transcription):
        """Calculate an overall quality score based on metrics and transcription."""
        score = 0
        max_score = 100
        details = []
        
        # Score signal level (0-25 points)
        signal_score = min(25, max(0, metrics['signal_level'] * 500))
        score += signal_score
        details.append(f"Signal Level: {signal_score:.1f}/25")
        
        # Score SNR (0-25 points)
        snr_score = min(25, max(0, metrics['snr']))
        score += snr_score
        details.append(f"SNR: {snr_score:.1f}/25")
        
        # Score dynamic range (0-25 points)
        dynamic_score = min(25, metrics['dynamic_range'] * 100)
        score += dynamic_score
        details.append(f"Dynamic Range: {dynamic_score:.1f}/25")
        
        # Score transcription quality (0-25 points)
        if transcription and len(transcription) > 0:
            words = transcription.split()
            transcription_score = min(25, len(words) * 2)  # 2 points per word, max 25
            score += transcription_score
            details.append(f"Transcription Quality: {transcription_score:.1f}/25")
        
        return {
            'total_score': score,
            'percentage': (score / max_score) * 100,
            'details': details,
            'metrics': metrics
        }

    def test_device(self, device_index):
        """Test the selected device and return quality metrics."""
        print(f"\nTesting device {device_index}...")
        print(f"Please speak continuously for {self.test_duration} seconds...")
        
        try:
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
            
            # Record audio
            audio_data = []
            start_time = time.time()
            
            while time.time() - start_time < self.test_duration:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data.append(np.frombuffer(data, dtype=np.int16))

            # Process audio data
            audio_data = np.concatenate(audio_data)
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Calculate metrics
            metrics = self.calculate_audio_metrics(audio_float)
            
            # Transcribe
            print("\nTranscribing test audio...")
            result = self.model.transcribe(audio_float, language="en")
            transcription = result["text"].strip()
            print(f"Transcribed text: {transcription}")
            
            # Calculate quality score
            quality_results = self.score_device(metrics, transcription)
            
            stream.stop_stream()
            stream.close()
            
            return True, quality_results, transcription
            
        except Exception as e:
            print(f"\nError testing device: {e}")
            return False, None, None

    def test_single_device(self, device_index):
        """Test a single device and display its results."""
        success, quality_results, transcription = self.test_device(device_index)
        
        if success:
            print("\n" + "="*50)
            print("DEVICE TEST RESULTS")
            print("="*50)
            print(f"Overall Score: {quality_results['percentage']:.1f}%")
            print("\nDetailed Scores:")
            for detail in quality_results['details']:
                print(f"  {detail}")
            print(f"\nTranscription: {transcription}")
            
            save = input("\nWould you like to save this device configuration? (yes/no): ").lower()
            if save.startswith('y'):
                self.save_config(device_index)
                print("\nConfiguration saved successfully!")
                return True
        else:
            print("\nDevice test failed. Please try another device.")
            return False

    def save_config(self, device_index):
        """Save the selected device index to config file while preserving other settings."""
        # Define audio-specific configuration keys
        audio_config_keys = {
            "audio_device_index",
            "sample_rate",
            "chunk_size",
            "channels"
        }
        
        # Create new audio configuration
        new_audio_config = {
            "audio_device_index": device_index,
            "sample_rate": self.sample_rate,
            "chunk_size": self.chunk_size,
            "channels": self.channels
        }
        
        # Read existing configuration if it exists
        existing_config = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    existing_config = json.load(f)
            except json.JSONDecodeError:
                print(f"\nWarning: Existing config file was corrupted. Creating new config.")
        
        # Update only audio-specific settings while preserving others
        final_config = existing_config.copy()
        for key, value in new_audio_config.items():
            final_config[key] = value
        
        # Save the updated configuration
        with open(self.config_path, 'w') as f:
            json.dump(final_config, f, indent=4)
            print(f"\nAudio configuration updated in {self.config_path}")

    def test_all_devices(self):
        """Run the optimal device test on all compatible devices."""
        print("\nScanning and testing all compatible audio devices...")
        valid_devices = []
        device_scores = {}
        
        # First pass: find valid devices
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0 and self.validate_device(device_info):
                valid_devices.append(device_info)
        
        if not valid_devices:
            print("No compatible input devices found!")
            return
        
        print("\nFound compatible devices. Let's test each one:")
        
        # Test each valid device
        for device_info in valid_devices:
            print(f"\n{'-'*50}")
            print(f"Testing device: {device_info['name']} (index: {device_info['index']})")
            success, quality_results, transcription = self.test_device(device_info['index'])
            
            if success:
                device_scores[device_info['index']] = {
                    'name': device_info['name'],
                    'score': quality_results,
                    'transcription': transcription
                }
        
        # Display results
        print("\n" + "="*50)
        print("DEVICE TEST RESULTS")
        print("="*50)
        
        sorted_devices = sorted(
            device_scores.items(),
            key=lambda x: x[1]['score']['total_score'],
            reverse=True
        )
        
        for device_index, data in sorted_devices:
            print(f"\nDevice: {data['name']} (index: {device_index})")
            print(f"Overall Score: {data['score']['percentage']:.1f}%")
            print("Detailed Scores:")
            for detail in data['score']['details']:
                print(f"  {detail}")
            print(f"Sample Transcription: {data['transcription']}")
            print("-"*50)
        
        # Recommend best device
        if sorted_devices:
            best_device = sorted_devices[0]
            print(f"\nRECOMMENDED DEVICE: {best_device[1]['name']} (index: {best_device[0]})")
            print(f"Score: {best_device[1]['score']['percentage']:.1f}%")
            
            use_recommended = input("\nWould you like to use the recommended device? (yes/no): ").lower()
            if use_recommended.startswith('y'):
                self.save_config(best_device[0])
            else:
                # Let user choose from tested devices
                while True:
                    try:
                        choice = int(input("\nEnter the index of the device you want to use (-1 to exit): "))
                        if choice == -1:
                            break
                        if choice in device_scores:
                            self.save_config(choice)
                            break
                        else:
                            print("Invalid device index. Please choose from the tested devices.")
                    except ValueError:
                        print("Please enter a valid number!")

    def run(self):
        """Main program loop with menu options."""
        while True:
            try:
                self.display_menu()
                choice = input("\nEnter your choice (1-4): ")
                
                if choice == '1':
                    self.list_all_devices()
                    # No need for input, automatically return to menu
                    time.sleep(1.5)  # Brief pause to let user read the device list
                    
                elif choice == '2':
                    devices = self.list_all_devices()
                    if not devices:
                        print("\nNo input devices found!")
                        time.sleep(1.5)
                        continue
                        
                    try:
                        device_index = int(input("\nEnter the device number to test (-1 to cancel): "))
                        if device_index == -1:
                            continue
                            
                        if not any(d['index'] == device_index for d in devices):
                            print("Invalid device number!")
                            time.sleep(1.5)
                            continue
                            
                        if not self.validate_device(self.p.get_device_info_by_index(device_index)):
                            print("\nThis device is not compatible with the required configuration:")
                            print(f"- Sample rate: {self.sample_rate} Hz")
                            print(f"- Channels: {self.channels}")
                            print("Please try another device.")
                            time.sleep(2)
                            continue
                            
                        self.test_single_device(device_index)
                        
                    except ValueError:
                        print("Please enter a valid number!")
                        time.sleep(1.5)
                        
                elif choice == '3':
                    self.test_all_devices()
                    
                elif choice == '4':
                    print("\nExiting...")
                    break
                    
                else:
                    print("\nInvalid choice. Please enter a number between 1-4.")
                    time.sleep(1.5)
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                continue

        self.p.terminate()

if __name__ == "__main__":
    print("Audio Device Selection and Testing Tool")
    print("======================================")
    print("This tool helps you select and test your microphone.")
    print("You can view devices, test individual devices, or")
    print("run an optimal test across all compatible devices.")
    
    tester = AudioDeviceTester()
    tester.run()