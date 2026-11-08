import difflib
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
import numpy as np
import requests
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI
from openwakeword.vad import VAD
from faster_whisper import WhisperModel

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

NAME = "diva"
MODEL = "gpt-6-luna"
TTS_MODEL = "gpt-4o-mini-tts"
VOICE = "shimmer"
VOICE_STYLE = "Bicara sesuai bahasa teksnya (Indonesia atau Inggris) dengan suara lembut, rendah, sedikit serak, tenang, pelan, hangat, dan santai."
TTS_RATE = 24000
APPS = {
    "cursor": "Cursor",
    "kursor": "Cursor",
    "chrome": "Google Chrome",
    "krom": "Google Chrome",
    "spotify": "Spotify",
}
OPEN_WORDS = {"buka", "bukain", "open", "jalankan"}
CLOSE_WORDS = {"tutup", "close", "keluar", "matikan", "quit"}
STOP_WORDS = {"off", "quit", "exit", "shutdown", "tidur", "bobo", "keluar"}
EXIT_WORDS = {"udah", "udahan", "makasih", "selesai", "dadah", "bye", "thanks", "done"}
REMEMBER_WORDS = {"inget", "ingat", "remember"}
LANGUAGES = ("id", "en")
SENTENCE_END = re.compile(r"(?<=[.!?])\s")
FIRST_CHUNK_MIN = 20
MEMORY_FILE = HERE / "memory.md"
MEMORY_LINES = 40
MUSIC_WORDS = {"lagu", "musik", "music", "spotify"}
MUSIC_ACTIONS = {
    "next": ("next track", "Lagu berikutnya"),
    "previous": ("previous track", "Lagu sebelumnya"),
    "pause": ("pause", None),
    "play": ("play", None),
    "volume_up": ("set sound volume to (sound volume + 20)", "Volume musik dinaikin"),
    "volume_down": ("set sound volume to (sound volume - 20)", "Volume musik diturunin"),
}
MUSIC_COMMANDS = [
    ({"next", "skip", "selanjutnya", "lewati", "ganti"}, "next"),
    ({"previous", "sebelumnya", "balik"}, "previous"),
    ({"pause", "jeda", "stop", "berhenti", "matikan", "matiin"}, "pause"),
    ({"gedein", "kerasin", "louder"}, "volume_up"),
    ({"kecilin", "pelanin", "quieter"}, "volume_down"),
    ({"play", "putar", "puter", "lanjut", "lanjutin", "nyalain"}, "play"),
]
PLAY_WORDS = {"play", "putar", "puter", "puterin", "setel", "setelin"}
SONG_FILLER = MUSIC_WORDS | PLAY_WORDS | {
    "di", "dong", "ya", "dari", "by", "the", "song", "songs", "lagunya", "tolong", "please", "can", "you", "bisa"
}
NOW_PLAYING ='if application "Spotify" is running then tell application "Spotify" to return (name of current track) & " oleh " & (artist of current track)'
HINT = "Diva, buka cursor. Diva, open spotify. Diva, next lagu. Diva, aku mau cerita. Diva, let's talk."
RATE, CHUNK = 16000, 480
PRE_ROLL, SILENCE_END, MAX_SECONDS, CHAT_TIMEOUT = 0.3, 1.0, 20, 15

RULES = """Kamu adalah Diva, asisten suara pribadi.
Jawabanmu akan dibacakan oleh text-to-speech, jadi jawab santai dengan bahasa yang sama seperti user (Indonesia, Inggris, atau campuran keduanya), 1 sampai 3 kalimat, tanpa markdown, emoji, daftar, atau URL.
Kalau user curhat, dengarkan dulu, validasi perasaannya, lalu tanya satu pertanyaan lanjutan. Jangan langsung memberi nasihat panjang.
Kamu bisa mengontrol app, Spotify, ingatan, dan dirimu sendiri lewat alat yang tersedia. Kalau user meminta aksi yang ada alatnya, panggil alatnya dan jangan menulis teks apa pun, karena alatnya akan memberi konfirmasi sendiri. Jangan bilang kamu tidak bisa melakukan sesuatu kalau alatnya ada.
Kamu bukan psikolog dan tidak boleh mendiagnosis. Kalau user menyebut ingin menyakiti diri, ingin mati, atau kondisinya terdengar sangat berat, dengan lembut sarankan menghubungi layanan kesehatan jiwa 119 ekstensi 8 atau orang terdekat, dan tetap temani dia.
Jangan pernah berpura-pura menjadi manusia."""

def tool(name, description, **properties):
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
        "strict": True,
    }


APP_NAMES = sorted(set(APPS.values()))
TOOLS = [
    tool("play_song", "Cari dan putar lagu tertentu di Spotify.",
         query={"type": "string", "description": "Judul lagu dan/atau nama penyanyi"}),
    tool("music", "Kontrol Spotify: play, pause, lagu berikutnya/sebelumnya, volume musik naik/turun, atau sebutkan lagu yang sedang diputar.",
         action={"type": "string", "enum": [*MUSIC_ACTIONS, "now_playing"]}),
    tool("open_app", "Buka aplikasi di Mac.", app={"type": "string", "enum": APP_NAMES}),
    tool("close_app", "Tutup aplikasi di Mac.", app={"type": "string", "enum": APP_NAMES}),
    tool("remember", "Simpan fakta yang user minta secara eksplisit untuk diingat atau diingatkan. Bukan pengingat berwaktu.",
         fact={"type": "string", "description": "Fakta singkat dalam bahasa user"}),
    tool("end_chat", "Akhiri mode ngobrol saat user pamit atau bilang sudah selesai ngobrol. Diva tetap menyala."),
    tool("shutdown", "Matikan Diva dan tutup terminalnya saat user minta Diva off, berhenti, quit, atau tutup terminal."),
]

vad = VAD()
stt = WhisperModel("small", compute_type="int8")
client = OpenAI()
SUMMARY_RULES = """Dari obrolan berikut, tulis 0 sampai 3 poin singkat tentang user yang berguna diingat untuk obrolan berikutnya: kejadian penting, perasaan, rencana, atau preferensi baru.
Satu poin per baris, diawali "- ". Jangan ulangi hal yang sudah ada di profil atau ingatan. Jangan tulis nama orang lain kecuali user sendiri yang menyebutnya. Kalau tidak ada yang penting, tulis KOSONG."""

history = deque(maxlen=12)


def say(text):
    print("Diva:", text)
    start = time.time()
    try:
        with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL, voice=VOICE, input=text.replace("*", ""), instructions=VOICE_STYLE, response_format="pcm"
        ) as response, sd.RawOutputStream(samplerate=TTS_RATE, channels=1, dtype="int16") as speaker:
            for i, chunk in enumerate(response.iter_bytes(4096)):
                if i == 0:
                    print(f"(suara mulai {time.time() - start:.1f}s)")
                speaker.write(chunk)
    except Exception as error:
        print("TTS error:", error)
        subprocess.run(["say", "-v", "Damayanti", text])


def flush(mic):
    if mic.read_available:
        mic.read(mic.read_available)


def record_utterance(mic, timeout=None):
    before = deque(maxlen=int(PRE_ROLL * RATE / CHUNK))
    frames, silent, waited = [], 0, 0
    while True:
        frame = mic.read(CHUNK)[0].flatten()
        talking = vad.predict(frame) > 0.5
        if not frames:
            before.append(frame)
            waited += 1
            if talking:
                frames = list(before)
            elif timeout and waited * CHUNK / RATE > timeout:
                return None
            continue
        frames.append(frame)
        silent = 0 if talking else silent + 1
        if silent * CHUNK / RATE > SILENCE_END or len(frames) * CHUNK / RATE > MAX_SECONDS:
            return np.concatenate(frames).astype("float32") / 32768


def hear(mic, timeout=None):
    audio = record_utterance(mic, timeout)
    if audio is None:
        return []
    segments, info = stt.transcribe(audio, vad_filter=True, initial_prompt=HINT)
    if info.language not in LANGUAGES:
        probs = dict(info.all_language_probs or [])
        language = max(LANGUAGES, key=lambda code: probs.get(code, 0))
        segments, info = stt.transcribe(audio, language=language, vad_filter=True, initial_prompt=HINT)
    text = " ".join(s.text for s in segments).strip()
    if text:
        print("Kamu:", text)
    return re.findall(r"\w+", text.lower())


def find_name(words):
    for i, word in enumerate(words):
        if word == NAME:
            return i
    return None


def open_app(app):
    subprocess.run(["open", "-a", app])
    say(f"Membuka {app}")


def close_app(app):
    subprocess.run(["osascript", "-e", f'quit app "{app}"'])
    say(f"Menutup {app}")


def run_app(words):
    closing = bool(CLOSE_WORDS & set(words))
    if not closing and not OPEN_WORDS & set(words):
        return False
    for word in words:
        match = difflib.get_close_matches(word, APPS, n=1, cutoff=0.7)
        if match:
            (close_app if closing else open_app)(APPS[match[0]])
            return True
    return False


def read(path):
    return path.read_text() if path.exists() else ""


def context():
    memory = "\n".join(read(MEMORY_FILE).splitlines()[-MEMORY_LINES:])
    now = datetime.now().strftime("%A, %d %B %Y, %H:%M")
    return f"# Tentang user\n{read(HERE / 'profile.md')}\n\n# Ingatan dari obrolan sebelumnya\n{memory}\n\n# Waktu sekarang\n{now}"


def remember(lines):
    if not MEMORY_FILE.exists():
        MEMORY_FILE.touch(mode=0o600)
    today = datetime.now().strftime("%Y-%m-%d")
    with MEMORY_FILE.open("a") as file:
        for line in lines:
            file.write(f"- {today}: {line.lstrip('- ').strip()}\n")


def save_session():
    if len(history) < 2:
        return history.clear()
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    try:
        response = client.responses.create(
            model=MODEL,
            instructions=f"{SUMMARY_RULES}\n\n{context()}",
            input=transcript,
            reasoning={"effort": "low"},
            store=False,
        )
        lines = [l for l in response.output_text.splitlines() if l.strip().startswith("-")]
    except Exception as error:
        return print("Memory error:", error)
    history.clear()
    if lines:
        remember(lines)
        print(f"(Diva nyimpen {len(lines)} ingatan baru)")


def chat(text):
    history.append({"role": "user", "content": text})
    start = time.time()
    reply, pending, spoken, calls = "", "", False, []
    try:
        stream = client.responses.create(
            model=MODEL,
            instructions=f"{RULES}\n\n{context()}",
            input=list(history),
            tools=TOOLS,
            reasoning={"effort": "low"},
            store=False,
            stream=True,
        )
        for event in stream:
            if event.type == "response.output_item.done" and event.item.type == "function_call":
                calls.append(event.item)
            if event.type != "response.output_text.delta":
                continue
            reply += event.delta
            pending += event.delta
            if spoken:
                continue
            end = next((m for m in SENTENCE_END.finditer(pending) if m.start() >= FIRST_CHUNK_MIN), None)
            if end:
                print(f"(kalimat pertama {time.time() - start:.1f}s)")
                say(pending[:end.start()].strip())
                pending, spoken = pending[end.end():], True
    except Exception as error:
        print("Error:", error)
        if not spoken:
            history.pop()
            say("Maaf, otakku lagi error.")
            return False
        reply, pending = reply[: len(reply) - len(pending)], ""
    if pending.strip():
        if not spoken:
            print(f"(jawaban siap {time.time() - start:.1f}s)")
        say(pending.strip())
    actions = [f"(aksi: {call.name} {call.arguments})" for call in calls]
    history.append({"role": "assistant", "content": " ".join([reply.strip(), *actions]).strip()})
    ends_chat = False
    for call in calls:
        print(f"(alat: {call.name} {call.arguments})")
        try:
            ends_chat = run_tool(call.name, json.loads(call.arguments)) or ends_chat
        except Shutdown:
            raise
        except Exception as error:
            print("Tool error:", error)
    return ends_chat


spotify_token = None


def find_track(query):
    global spotify_token
    if not spotify_token or spotify_token[1] < time.time():
        auth = (os.environ["SPOTIFY_CLIENT_ID"], os.environ["SPOTIFY_CLIENT_SECRET"])
        body = requests.post(
            "https://accounts.spotify.com/api/token", data={"grant_type": "client_credentials"}, auth=auth, timeout=10
        ).json()
        spotify_token = (body["access_token"], time.time() + body["expires_in"] - 60)
    response = requests.get(
        "https://api.spotify.com/v1/search",
        params={"q": query, "type": "track", "limit": 1},
        headers={"Authorization": f"Bearer {spotify_token[0]}"},
        timeout=10,
    )
    response.raise_for_status()
    items = response.json()["tracks"]["items"]
    return items[0] if items else None


def play_song(query):
    if not os.environ.get("SPOTIFY_CLIENT_ID"):
        subprocess.run(["open", f"spotify:search:{quote(query)}"])
        return say(f"Gue cariin {query} di Spotify, pilih lagunya ya.")
    try:
        track = find_track(query)
    except Exception as error:
        print("Spotify error:", error)
        return say("Maaf, gue gak bisa nyari lagunya sekarang.")
    if not track or not re.fullmatch(r"spotify:track:[A-Za-z0-9]+", track["uri"]):
        return say(f"Gue gak nemu lagu {query}.")
    say(f"Muter {track['name']} dari {track['artists'][0]['name']}.")
    subprocess.run(["osascript", "-e", f'tell application "Spotify" to play track "{track["uri"]}"'])


def control_music(words):
    start = next((i for i, w in enumerate(words) if w in PLAY_WORDS), None)
    query = [w for w in words[start + 1:] if w not in SONG_FILLER] if start is not None else []
    if query:
        play_song(" ".join(query))
        return True
    if not MUSIC_WORDS & set(words):
        return False
    if {"apa", "judul"} & set(words):
        music_action("now_playing")
        return True
    for keywords, action in MUSIC_COMMANDS:
        if keywords & set(words):
            music_action(action)
            return True
    return False


def music_action(action):
    if action == "now_playing":
        playing = subprocess.run(["osascript", "-e", NOW_PLAYING], capture_output=True, text=True).stdout.strip()
        return say(f"Ini {playing}" if playing else "Lagi gak ada lagu yang diputar")
    script, reply = MUSIC_ACTIONS[action]
    subprocess.run(["osascript", "-e", f'tell application "Spotify" to {script}'])
    if reply:
        say(reply)
    else:
        print("Diva:", f"({action})")


class Shutdown(Exception):
    pass


def try_shutdown(words):
    stop_session = "stop" in words and {"session", "sesi", "terminal"} & set(words)
    if (words and words[0] in STOP_WORDS) or stop_session:
        raise Shutdown
    return False


def close_terminal_window():
    try:
        tty = os.ttyname(sys.stdin.fileno())
    except OSError:
        return
    lines = [
        "delay 1",
        'tell application "Terminal"',
        "repeat with w in windows",
        "repeat with t in tabs of w",
        f'if tty of t is "{tty}" then',
        "close w",
        "return",
        "end if",
        "end repeat",
        "end repeat",
        "end tell",
    ]
    subprocess.Popen(["osascript", *[arg for line in lines for arg in ("-e", line)]], start_new_session=True)


def run_tool(name, args):
    if name == "shutdown":
        raise Shutdown
    if name == "end_chat":
        say("Oke, panggil aja kalau butuh.")
        return True
    if name == "remember":
        remember([args["fact"]])
        say("Oke, gue inget.")
    elif name == "open_app":
        open_app(args["app"])
    elif name == "close_app":
        close_app(args["app"])
    elif name == "music":
        music_action(args["action"])
        return args["action"] == "play"
    elif name == "play_song":
        play_song(args["query"])
        return True
    return False


def local_command(words):
    return run_app(words) or control_music(words) or try_remember(words) or try_shutdown(words)


def try_remember(words):
    if not words or words[0] not in REMEMBER_WORDS:
        return False
    rest = words[2:] if words[1:2] == ["ya"] else words[1:]
    if not rest:
        say("Inget apa?")
        return True
    remember([" ".join(rest)])
    say("Oke, gue inget.")
    return True


def main():
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=CHUNK) as mic:
        print("Diva siap. Bilang: Diva, buka cursor  /  Diva, aku mau cerita")
        in_chat = False
        while True:
            words = hear(mic, CHAT_TIMEOUT if in_chat else None)
            if in_chat:
                i = find_name(words)
                if not words:
                    in_chat = False
                    print("(mode ngobrol selesai, panggil Diva lagi)")
                    save_session()
                elif len(words) <= 3 and EXIT_WORDS & set(words):
                    in_chat = False
                    say("Oke, panggil aja kalau butuh.")
                    save_session()
                elif local_command(words[i + 1:]) if i is not None else control_music(words):
                    in_chat = False
                    save_session()
                elif chat(" ".join(words)):
                    in_chat = False
                    save_session()
                flush(mic)
                continue
            i = find_name(words)
            if i is None:
                continue
            command = words[i + 1:]
            if not command:
                say("Ya?")
                flush(mic)
                command = hear(mic, CHAT_TIMEOUT)
                if not command:
                    continue
            if not local_command(command):
                in_chat = not chat(" ".join(command))
            flush(mic)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        save_session()
        print("\nDiva off.")
    except Shutdown:
        say("Oke, Diva pamit dulu. Sampai nanti.")
        save_session()
        print("Diva off.")
        close_terminal_window()
