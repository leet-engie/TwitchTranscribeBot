# validate_device.py

import sys
import pyaudio

def main():
    if len(sys.argv) != 4:
        print("Usage: validate_device.py <device_index> <sample_rate> <channels>")
        sys.exit(1)
    
    device_index = int(sys.argv[1])
    sample_rate = int(sys.argv[2])
    channels = int(sys.argv[3])
    
    p = pyaudio.PyAudio()
    try:
        # Attempt to open the stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024
        )
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Supported")
    except Exception as e:
        print("Not Supported")
        p.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()
