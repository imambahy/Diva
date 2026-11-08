import difflib
import re
import subprocess
import numpy as np
import sounddevice as sd
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel

APPS = {
    "cursor": "Cursor",
    "kursor": "Cursor",
    "chrome": "Google Chrome",
    "krom": "Google Chrome",
    "spotify": "Spotify",
}
HINT = "buka cursor, buka chrome, buka spotify"
RATE, CHUNK = 16000, 1280

openwakeword.utils.download_models(model_names=["hey_jarvis"])
wake = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx", vad_threshold=0.5)
stt = WhisperModel("small", compute_type="int8")


def say(text):
    subprocess.run(["say", "-v", "Damayanti", text])


def listen(mic, seconds=4):
    if mic.read_available:
        mic.read(mic.read_available)
    frames = [mic.read(CHUNK)[0] for _ in range(int(seconds * RATE / CHUNK))]
    return np.concatenate(frames).flatten().astype("float32") / 32768


def handle(text):
    print("Dengar:", text)
    if not text.strip():
        return say("Gak kedengeran")
    for word in re.findall(r"\w+", text.lower()):
        match = difflib.get_close_matches(word, APPS, n=1, cutoff=0.7)
        if match:
            app = APPS[match[0]]
            subprocess.run(["open", "-a", app])
            return say(f"Membuka {app}")
    say("Maaf, belum ngerti")


with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=CHUNK) as mic:
    print("Diva siap. Bilang: hey jarvis")
    while True:
        frame, _ = mic.read(CHUNK)
        score = wake.predict(frame.flatten())["hey_jarvis"]
        if score > 0.05:
            print(f"skor: {score:.2f}")
        if score > 0.35:
            say("Ya?")
            segments, _ = stt.transcribe(listen(mic), language="id", vad_filter=True, initial_prompt=HINT)
            handle(" ".join(s.text for s in segments))
            wake.reset()
