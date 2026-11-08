# Diva

A personal voice assistant that runs on macOS. Say "Diva, ..." and it opens apps, controls Spotify, remembers things, or just talks with you, in Indonesian or English.

Built as a learning project for AI automation: voice activity detection, local speech-to-text, LLM chat with function calling, streaming text-to-speech, and long-term memory.

## Features

- **Wake by name.** Diva only reacts to sentences that contain "Diva". Lyrics, videos, and other conversations are ignored.
- **Bilingual.** Detects Indonesian or English per sentence and answers in the same language.
- **Chat mode.** After the first "Diva, ...", keep talking without repeating the name. It ends when you say "udah" / "thanks", or after 15 seconds of silence.
- **App control.** Opens and closes whitelisted apps (Cursor, Chrome, Spotify).
- **Spotify.** Play a specific song, play/pause, next/previous, volume, "what song is this?".
- **Memory.** Remembers facts you ask it to, and summarizes each conversation into short notes for next time.
- **Function calling.** Free-form requests ("can you turn down the volume?") are routed by GPT to a fixed set of tools.
- **Voice shutdown.** "Diva, off" saves memory and closes its own Terminal window.

## How it works

```
mic ──> Silero VAD (detects when you speak)
          │
          ▼
        faster-whisper "small", local (speech to text, id/en)
          │
          ▼
        contains "diva"? ── no ──> ignore
          │ yes
          ▼
        keyword match? ── yes ──> run command (fast path)
          │ no
          ▼
        OpenAI chat (streamed, with tools) ──> tool call, or text reply
          │
          ▼
        OpenAI TTS (streamed PCM) ──> speaker
```

Speech-to-text runs locally. Only the transcribed text of what you say to Diva, plus `profile.md` and recent `memory.md` lines, is sent to OpenAI.

## Requirements

- macOS on Apple Silicon
- Python 3.12 or newer (developed on 3.14)
- An OpenAI API key
- Optional: a Spotify developer app (Client ID and Secret) to play songs directly

## Setup

```bash
cd ~/diva
python3 -m venv .venv
.venv/bin/pip install openwakeword faster-whisper sounddevice openai python-dotenv requests
```

Create `.env` (keep it private, it is in `.gitignore`):

```bash
touch .env && chmod 600 .env
```

```
OPENAI_API_KEY=sk-...
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
```

The Spotify lines are optional. Without them, "Diva, play ..." opens Spotify search results instead of playing the song directly.

Create `profile.md` with a few lines about yourself (name, how you like to be talked to, what you're working on). Diva reads it on every chat. It is in `.gitignore` because it is personal.

Run:

```bash
.venv/bin/python diva.py
```

The first run downloads the Whisper and VAD models. macOS will ask for microphone access, and later for permission to control Spotify and Terminal.

## Launch with a shortcut

Create a small app that starts Diva in Terminal:

```bash
osacompile -o /Applications/Diva.app \
  -e 'tell application "Terminal"' \
  -e 'do script "cd ~/diva && exec .venv/bin/python diva.py"' \
  -e 'activate' \
  -e 'end tell'
plutil -insert CFBundleIdentifier -string id.imam.diva /Applications/Diva.app/Contents/Info.plist
codesign --force --sign - /Applications/Diva.app
```

Then in the Shortcuts app: new shortcut "Diva", action **Open App → Diva**, and set a keyboard shortcut (for example ⌃⌥D) in the shortcut's details.

## Voice commands

| Say | Result |
|---|---|
| "Diva, buka cursor" / "Diva, open spotify" | Opens the app |
| "Diva, tutup spotify" / "Diva, quit spotify" | Closes the app |
| "Diva, play Gravity by John Mayer" | Searches and plays the song |
| "Diva, next lagu" / "pause musik" / "lagu apa ini?" | Spotify control |
| "Diva, inget ya ..." / "Diva, remember ..." | Saves a fact to memory |
| "Diva, aku mau cerita" | Starts chat mode |
| "udah makasih" / "thanks" | Ends chat mode |
| "Diva, off" / "Diva, stop the session" | Saves memory and quits |

Anything else after "Diva" goes to GPT, which can answer or call one of the tools.

## Configuration

At the top of `diva.py`:

| Setting | Purpose |
|---|---|
| `MODEL` | Chat model |
| `TTS_MODEL`, `VOICE`, `VOICE_STYLE` | Voice and speaking style. Try voices with `.venv/bin/python voices.py` |
| `APPS` | Whitelist of apps Diva may open or close |
| `CHAT_TIMEOUT` | Seconds of silence before chat mode ends |
| `MEMORY_LINES` | How many recent memory lines are sent with each chat |

## Files

| File | Purpose |
|---|---|
| `diva.py` | The assistant |
| `voices.py` | Plays sample sentences in several OpenAI voices |
| `diva_jarvis.py` | First version, using the "hey jarvis" wake word model |
| `profile.md` | About you (private, not committed) |
| `memory.md` | Diva's notes from past conversations (private, not committed) |
| `.env` | API keys (private, not committed) |

## Privacy and safety

- API keys stay in `.env`, readable only by your user.
- Chat requests use `store=False`. Audio recordings are never saved or uploaded.
- `profile.md` and `memory.md` are plain text files. Edit or delete lines anytime.
- GPT can only call the tools defined in `diva.py`. There is no tool that runs arbitrary shell commands.
- Diva is not a therapist. Its instructions tell it to suggest professional help (Indonesia: 119 ext. 8) when a conversation sounds serious.
- Only use a cloned voice from someone who has given explicit consent.

## Troubleshooting

| Problem | Fix |
|---|---|
| `command not found: python` | Use `.venv/bin/python diva.py` |
| Diva never reacts | System Settings → Privacy & Security → Microphone → allow Terminal |
| Spotify commands do nothing | Allow "Terminal wants to control Spotify" when macOS asks |
| `TTS error: ...` | Diva falls back to the macOS `say` voice. Check the API key and model name |
| Whisper hears words you didn't say | Usually background noise or music. It is ignored unless it contains "Diva" |
