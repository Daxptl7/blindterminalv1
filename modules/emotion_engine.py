"""
emotion_engine.py — BlindAssist Project (OPTIMIZED)
=====================================================
Real-time emotion detection from audio buffer.
No disk writes. Processes while recording.
"""

import sys
import signal
import logging
import time
import numpy as np
import io
import threading
import queue

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "emotion.log"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("EmotionModule")

# ── LAZY LOADING ────────────────────────────────────────────
_librosa = None
_sklearn = None

def _ensure_libs():
    global _librosa, _sklearn
    if _librosa is None:
        import librosa
        _librosa = librosa
    if _sklearn is None:
        import sklearn
        _sklearn = sklearn
    return _librosa, _sklearn

# ── FEATURE EXTRACTION (from numpy buffer, no disk) ─────────
def extract_features_from_buffer(audio_buffer: np.ndarray, sr: int = 22050):
    """Extract emotion features directly from audio buffer."""
    librosa, _ = _ensure_libs()
    
    if len(audio_buffer) < sr * 0.5:  # Need 0.5s minimum
        return None
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_buffer, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)
    
    # Spectral features
    cent = librosa.feature.spectral_centroid(y=audio_buffer, sr=sr)
    cent_mean = np.mean(cent)
    
    # Energy
    rms = librosa.feature.rms(y=audio_buffer)
    rms_mean = np.mean(rms)
    
    # Zero crossing rate (hesitation indicator)
    zcr = librosa.feature.zero_crossing_rate(audio_buffer)
    zcr_mean = np.mean(zcr)
    
    return np.hstack([mfcc_means, cent_mean, rms_mean, zcr_mean]).reshape(1, -1)

# ── REAL-TIME CLASSIFIER ────────────────────────────────────
class RealtimeEmotionDetector:
    def __init__(self):
        self.buffer = np.array([], dtype=np.float32)
        self.sr = 22050
        self.chunk_size = 1024
        self.lock = threading.Lock()
        self.last_emotion = "CALM"
        self.consecutive_stress = 0
        self._running = False
        self._thread = None
        
    def start(self):
        """Start background analysis thread."""
        self._running = True
        self._thread = threading.Thread(target=self._analyze_loop, daemon=True)
        self._thread.start()
        
    def feed_chunk(self, chunk: np.ndarray):
        """Feed audio chunk from microphone."""
        with self.lock:
            self.buffer = np.concatenate([self.buffer, chunk])
            # Keep last 3 seconds
            max_samples = self.sr * 3
            if len(self.buffer) > max_samples:
                self.buffer = self.buffer[-max_samples:]
    
    def _analyze_loop(self):
        """Background: analyze every 2 seconds."""
        while self._running:
            time.sleep(2.0)
            self._analyze()
    
    def _analyze(self):
        """Run classification on current buffer."""
        with self.lock:
            if len(self.buffer) < self.sr * 1.0:
                return
            audio = self.buffer.copy()
        
        features = extract_features_from_buffer(audio, self.sr)
        if features is None:
            return
        
        # Simple rule-based for speed (replace with SVM if trained)
        rms = features[0, -2]
        zcr = features[0, -1]
        centroid = features[0, -3]
        
        if rms > 0.1 and zcr > 0.15:
            emotion = "STRESSED"
        elif rms < 0.03 and zcr < 0.05:
            emotion = "CONFUSED"
        else:
            emotion = "CALM"
        
        if emotion != "CALM":
            self.consecutive_stress += 1
        else:
            self.consecutive_stress = 0
        
        self.last_emotion = emotion
        logger.info(f"Emotion: {emotion} (stress_count={self.consecutive_stress})")
    
    def should_slow_down(self) -> bool:
        return self.consecutive_stress >= 2
    
    def get_emotion(self) -> str:
        return self.last_emotion
    
    def reset(self):
        self.consecutive_stress = 0
        self.last_emotion = "CALM"
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)

# ── GLOBAL INSTANCE ─────────────────────────────────────────
_detector = RealtimeEmotionDetector()

def start():
    _detector.start()

def feed_chunk(chunk: np.ndarray):
    _detector.feed_chunk(chunk)

def should_slow_down() -> bool:
    return _detector.should_slow_down()

def get_emotion() -> str:
    return _detector.get_emotion()

def reset():
    _detector.reset()

if __name__ == '__main__':
    import time
    signal.signal(signal.SIGINT, lambda s, f: (_detector.stop(), sys.exit(0)))
    
    print("Emotion Engine — feeding synthetic audio...")
    start()
    
    # Simulate audio feed
    t = np.linspace(0, 5, 5 * 22050)
    for i in range(0, len(t), 1024):
        chunk = 0.1 * np.sin(2 * np.pi * 200 * t[i:i+1024]).astype(np.float32)
        feed_chunk(chunk)
        time.sleep(0.046)  # Real-time pace
    
    time.sleep(3)
    print(f"Final emotion: {get_emotion()}")
    print(f"Should slow down: {should_slow_down()}")
    _detector.stop()