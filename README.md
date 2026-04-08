# Discord Voice Bot Prototype

Minimal end-to-end test harness for a Discord voice bot with a hybrid Rust + Python architecture:

Rust voice gateway/receiver
-> buffered PCM utterance chunks
-> Python FastAPI STT + wake-word gate + TTS service
-> Rust playback back into the same voice channel

## Architecture Summary

- `rust-bot/`
  - Connects to Discord with `serenity`
  - Owns voice join/leave, receive, and playback with `songbird`
  - Buffers short utterances per SSRC using a simple energy + silence heuristic
  - Resolves SSRCs back to Discord users and maintains a per-guild speaker registry
  - Sends finalized WAV chunks to the local Python service over HTTP
- `python-service/`
  - Exposes a small FastAPI service
  - Runs local STT with `faster-whisper`
  - Only responds when the transcript explicitly addresses the bot
  - Synthesizes reply audio with AWS Polly

## File Tree

```text
.
├── .env.example
├── .gitignore
├── README.md
├── python-service
│   ├── app.py
│   ├── logic.py
│   └── requirements.txt
└── rust-bot
    ├── Cargo.toml
    └── src
        ├── audio.rs
        ├── main.rs
        ├── python_client.rs
        └── speaker_registry.rs
```

## Why These Libraries

- `serenity`: straightforward Rust Discord bot client with stable event handling.
- `songbird`: practical Rust voice stack with both send and receive support. Its `receive` feature exposes decoded audio via `VoiceTick`, which is exactly what this prototype needs.
- `FastAPI`: the simplest clean local HTTP boundary between Rust and Python.
- `faster-whisper`: easy local STT for a prototype, with decent CPU performance and very small integration code.
- `AWS Polly`: managed TTS with MP3 output and access to the British English `Brian` voice.

## Prerequisites

- Python 3.11+
- Rust 1.78+ with Cargo
- A Discord server where you can add a bot
- On Windows:
  - Visual Studio C++ Build Tools
  - `cmake`
  - This helps Songbird/audiopus build correctly
- Internet access on first run for:
  - `pip install`
  - Cargo dependency download
  - `faster-whisper` model download
  - AWS Polly voice synthesis
  - AWS credentials configured for Polly access

## Discord Bot Setup

1. Open the Discord Developer Portal: <https://discord.com/developers/applications>
2. Create a new application.
3. Open the `Bot` tab and create a bot user.
4. Copy the bot token.
5. Under `Privileged Gateway Intents`, enable:
   - `MESSAGE CONTENT INTENT`
6. In the bot invite permissions, include at least:
   - `View Channels`
   - `Send Messages`
   - `Read Message History`
   - `Connect`
   - `Speak`
7. Invite the bot with the `bot` scope.

Required gateway intents in the code:

- `GUILDS`
- `GUILD_MESSAGES`
- `MESSAGE_CONTENT`
- `GUILD_VOICE_STATES`

## Environment Configuration

Copy the example env file and fill in your values:

```powershell
Copy-Item .env.example .env
```

Important variables:

- `DISCORD_TOKEN`
- `PYTHON_SERVICE_URL`
- `FASTER_WHISPER_MODEL`
- `AWS_REGION`
- `POLLY_VOICE_ID`
- `VOICE_ENERGY_THRESHOLD`
- `VOICE_SILENCE_FRAMES`

See [.env.example](/c:/Users/jacksal1/Desktop/Voice Agent Frontend/Discord-Voice-Frontend/.env.example) for the full list.

Put your real AWS credentials in your local `.env` file, not in `.env.example`.
`.env` is already gitignored in this repo.

`boto3` will read them through the standard AWS SDK credential chain, such as:

- `aws configure`
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
- `AWS_SESSION_TOKEN` for temporary credentials
- an AWS profile exposed through your shell environment

## Install Python Service

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r python-service\requirements.txt
```

## Run Python Service

```powershell
.venv\Scripts\Activate.ps1
python python-service\app.py
```

Expected startup behavior:

- FastAPI starts on `127.0.0.1:8000` by default
- `faster-whisper` loads the configured model
- `/health` returns service status

## Run Rust Bot

```powershell
cargo run --manifest-path rust-bot\Cargo.toml
```

Expected startup behavior:

- bot logs in
- Songbird voice manager is registered
- the bot waits for text commands

## Discord Commands

- `!join`
  - Bot joins the sender's current voice channel
- `!leave`
  - Bot disconnects and removes the voice session
- `!status`
  - Shows whether the bot is connected to a voice channel
- `!pingvoice`
  - Calls the Python service `/health` endpoint

## Speaker Identity And Prototype Diarization

This prototype does not do offline diarization over a mixed channel recording.

Instead, "diarization" here means:

- Songbird receives decoded audio per SSRC
- Rust keeps an SSRC -> Discord user mapping from `SpeakingStateUpdate`
- Rust keeps a per-guild speaker registry with:
  - Discord user id
  - username and best available display name
  - latest SSRC and SSRC history
  - first-seen and last-seen timestamps
- Each finalized utterance is sent to Python with speaker metadata

If an SSRC cannot be mapped back to a Discord user in time, Rust falls back to a stable placeholder such as `unknown:<ssrc>` and logs the resolution source.

Speaker registry snapshots are written to `.runtime/guild-<guild_id>/speaker-registry.json`.

## Wake Names

The bot only replies when the transcript contains one of these wake names:

- `butler`
- `monkey`
- `monkey butler`

Matching is case-insensitive and normalized across simple punctuation and spacing differences. If an utterance is not addressed, Python returns a no-op response and Rust does not enqueue playback.

Edge cases handled:

- `!join` from a user not in voice
- `!join` while already connected
- `!leave` while not connected

## End-to-End Test Procedure

1. Start the Python service.
2. Start the Rust bot.
3. Join a Discord voice channel yourself.
4. Send `!join` in a text channel the bot can read.
5. Watch logs for:
   - bot connected
   - joined voice channel
   - Songbird driver connected
6. Speak a short phrase like:
   - `butler hello`
   - `monkey, what time is it`
   - `monkey butler say test successful`
7. Confirm Rust logs show:
   - speaking SSRC mapped to a Discord user
   - receiving audio frames
   - speech chunk finalized
   - transcript with speaker metadata and addressed / ignored outcome
   - selected response text only when addressed
   - audio queued to Discord only when addressed
8. Confirm Python logs show:
   - STT request received
   - transcript text and speaker metadata
   - ignored reason for unaddressed speech or selected reply for addressed speech
   - Polly TTS generated only when addressed
9. Listen for the reply in the voice channel.
10. Send `!leave`.
11. Confirm clean disconnect in logs.

## Prototype Behavior

Example flow:

1. User sends `!join`
2. Bot joins the voice channel
3. User says `butler, what time is it`
4. Rust buffers a short utterance, resolves the speaker from SSRC, and posts speaker-aware metadata to Python
5. Python transcribes the utterance and checks whether it addressed the bot
6. Intent logic selects a reply only when the wake name was present
7. Python generates MP3 TTS only for addressed utterances
8. Rust queues the MP3 into the Songbird call only when `should_respond = true`
9. Bot speaks the reply
10. User sends `!leave`

## Logging and Debugging

Rust logs include:

- bot connected
- joined channel
- speaking SSRC mapped to user
- speaker registry updates
- receiving audio frames
- speech chunk finalized
- transcript text with Discord speaker metadata
- addressed / ignored decision
- reply text
- TTS audio queued only for addressed utterances
- disconnect / shutdown

Python logs include:

- health checks
- chunk receive size
- transcript text with speaker metadata
- ignored reason when wake word was missing
- chosen reply text for addressed speech
- Polly TTS generation complete only for addressed speech

To increase Rust log detail:

```powershell
$env:RUST_LOG="rust_bot=debug,info"
cargo run --manifest-path rust-bot\Cargo.toml
```

## Known Limitations

- TTS uses AWS Polly, so TTS is not fully offline even though STT is local.
- The first `faster-whisper` run downloads the configured model.
- Utterance segmentation is intentionally simple and may miss very quiet speakers.
- Speaker identity is based on Discord/Songbird SSRC mapping, not biometric voice matching.
- There is no advanced mixed-audio diarization or conversation memory.
- The bot is driven by text commands, not slash commands.
- Multiple people speaking at exactly the same time can still produce imperfect chunking.

## Rough Edges Worth Knowing

- This is a prototype, so short temp audio files are written to `.runtime/`.
- Speaker handling is keyed by SSRC, mapped back to Discord users through `SpeakingStateUpdate`, and persisted as a lightweight per-guild registry.
- The receive path is designed to stay light by doing only chunk buffering in the Songbird event handler and offloading HTTP/TTS work to spawned tasks.
- The current speaker registry is intended for continuity and observability, and is the extension point for future embedding or fingerprint-based matching.

## Source Files

Main Rust bot entrypoint: [rust-bot/src/main.rs](/c:/Users/jacksal1/Desktop/Voice Agent Frontend/Discord-Voice-Frontend/rust-bot/src/main.rs)

Rust audio helper code: [rust-bot/src/audio.rs](/c:/Users/jacksal1/Desktop/Voice Agent Frontend/Discord-Voice-Frontend/rust-bot/src/audio.rs)

Rust Python client: [rust-bot/src/python_client.rs](/c:/Users/jacksal1/Desktop/Voice Agent Frontend/Discord-Voice-Frontend/rust-bot/src/python_client.rs)

Python FastAPI service: [python-service/app.py](/c:/Users/jacksal1/Desktop/Voice Agent Frontend/Discord-Voice-Frontend/python-service/app.py)
