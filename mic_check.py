import pyaudio
p = pyaudio.PyAudio()
try:
    info = p.get_default_input_device_info()
    print("Default input device:", info.get('name'))
    print("Native default sample rate:", info.get('defaultSampleRate'))
    print("Max input channels:", info.get('maxInputChannels'))
except Exception as e:
    print("Could not get default input device:", e)
print()
print("All input-capable devices:")
for i in range(p.get_device_count()):
    d = p.get_device_info_by_index(i)
    if d.get('maxInputChannels', 0) > 0:
        print(f"  [{i}] {d['name']}  defaultSampleRate={d['defaultSampleRate']}")
p.terminate()
