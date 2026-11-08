import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")
client = OpenAI()

TEXT = "Hai Imam, gue Diva. Hari ini capek ya? Cerita aja pelan-pelan, gue dengerin."
STYLE = "Bicara bahasa Indonesia dengan suara lembut, rendah, sedikit serak, tenang, pelan, hangat, dan santai."
VOICES = sys.argv[1:] or ["shimmer", "sage", "coral", "nova", "marin"]

for voice in VOICES:
    print("Suara:", voice)
    path = HERE / f".sample-{voice}.mp3"
    try:
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts", voice=voice, input=TEXT, instructions=STYLE
        ) as response:
            response.stream_to_file(path)
        subprocess.run(["afplay", str(path)])
    except Exception as error:
        print("  Error:", error)
