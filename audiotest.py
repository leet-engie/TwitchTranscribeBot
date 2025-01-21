import pyaudio
import numpy as np
import time

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()

# Get all input devices
input_devices = []
for i in range(p.get_device_count()):
    dev_info = p.get_device_info_by_index(i)
    if dev_info['maxInputChannels'] > 0:
        input_devices.append((i, dev_info['name']))
        print(f"Found input device {i}: {dev_info['name']}")

print("\nTesting each input device for 5 seconds...")

for device_id, name in input_devices:
    try:
        print(f"\nTesting device {device_id}: {name}")
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       input_device_index=device_id,
                       frames_per_buffer=CHUNK)
        
        # Test for 5 seconds
        start_time = time.time()
        max_volume = 0
        min_volume = float('inf')
        
        while time.time() - start_time < 5:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()
            max_volume = max(max_volume, volume)
            min_volume = min(min_volume, volume)
            print(f"\rVolume range: {min_volume:.0f} - {max_volume:.0f}", end='', flush=True)
            
        print(f"\nDevice {device_id} volume range: {min_volume:.0f} - {max_volume:.0f}")
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"Error testing device {device_id}: {e}")

p.terminate()