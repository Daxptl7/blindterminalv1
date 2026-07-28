import importlib
libs = ["cv2","pytesseract","PIL","numpy","pyttsx3","pygame",
        "speech_recognition","openai","pyaudio","serial","pynmea2",
        "mediapipe","librosa","sklearn","soundfile","ultralytics",
        "requests","groq","google.genai","faiss","dotenv"]
failed = []
for lib in libs:
    try:
        importlib.import_module(lib)
        print(f"[ OK ] {lib}")
    except ImportError as e:
        print(f"[FAIL] {lib} -> {e}")
        failed.append(lib)
print("ALL OK" if not failed else f"{len(failed)} missing")
