# Discord Voice Bot Prototype

Minimal end-to-end test harness for a Discord voice bot with a hybrid Rust + Python architecture:

Rust voice gateway/receiver
-> buffered PCM utterance chunks
-> Python FastAPI STT + conversation memory + OpenAI reply generation + TTS service
-> Rust playback back into the same voice channel

## Architecture Summary

- `rust-bot/`
  - Connects to Discord with `serenity`
  - Owns voice join/leave, receive, and playback with `songbird`
  - Buffers short utterances per SSRC using a simple energy + silence heuristic
  - Resolves SSRCs back to Discord users and maintains a per-guild speaker registry
  - Sends finalized WAV chunks plus speaker/call metadata to the local Python service over HTTP
- `python-service/`
  - Exposes a small FastAPI service
  - Runs local STT with `faster-whisper`
  - Maintains bounded per-guild conversation memory with speaker attribution
  - Builds a structured prompt with guild/channel/participant awareness
  - Generates replies through an LLM client wrapper
  - Synthesizes reply audio with AWS Polly

## File Tree

```text
.
├── .env.example
├── .gitignore
├── README.md
├── python-service
│   ├── app.py
│   ├── conversation_store.py
│   ├── llm_client.py
│   ├── logic.py
│   ├── prompt_builder.py
│   ├── prompt_config.py
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
- `openai`: straightforward hosted LLM integration with a clean provider boundary for future swaps.
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
  - OpenAI API replies
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
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `AWS_REGION`
- `POLLY_VOICE_ID`
- `VOICE_ENERGY_THRESHOLD`
- `VOICE_SILENCE_FRAMES`

See [.env.example](/mnt/c/Users/aljac/Desktop/Butler Discord Frontend/Discord-Voice-Frontend/.env.example) for the full list.

Put your real AWS credentials in your local `.env` file, not in `.env.example`.
`.env` is already gitignored in this repo.

OpenAI configuration is also local-only. Set at least:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`

Optional OpenAI tuning:

- `OPENAI_BASE_URL`
- `OPENAI_TIMEOUT_SECONDS`
- `OPENAI_MAX_OUTPUT_TOKENS`

Conversation-memory tuning:

- `MAX_HISTORY_MESSAGES`

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
- the OpenAI-backed LLM client is configured from environment
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

## Prompt And Personality

Bot personality and behavioral instructions live in [python-service/prompt_config.py](/mnt/c/Users/aljac/Desktop/Butler Discord Frontend/Discord-Voice-Frontend/python-service/prompt_config.py).

That file is the dedicated place to customize:

- personality and role
- tone and response style
- multi-user conversation behavior
- ambiguity handling
- voice-response constraints

## Conversation Memory

Python owns conversation memory because it already owns prompt assembly and reply generation.

- Memory is kept per guild in-process only.
- Each stored message records:
  - timestamp
  - speaker id
  - speaker name
  - role (`user` or `assistant`)
  - text
- Memory is bounded by `MAX_HISTORY_MESSAGES`.

This keeps the Rust voice path simple while giving the LLM speaker-attributed context.

## Wake Names And Reply Gating

The bot still recognizes these wake names:

- `butler`
- `monkey`
- `monkey butler`
- `clanker`

Matching is case-insensitive and normalized across simple punctuation and spacing differences.

Reply behavior is now:

- in a one-on-one voice session, the bot can respond without a wake name
- in calls with multiple human participants, the utterance must include a wake name
- otherwise return a no-op response and do not enqueue playback

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
   - `butler who was Bob talking about`
   - `and what about tomorrow`
7. Confirm Rust logs show:
   - speaking SSRC mapped to a Discord user
   - receiving audio frames
   - speech chunk finalized
   - transcript with speaker metadata plus guild/channel/participant context
   - addressed / follow-up / ignored outcome
   - selected response text from the Python service
   - audio queued to Discord only when `should_respond = true`
8. Confirm Python logs show:
   - STT request received
   - transcript text and speaker metadata
   - conversation-memory updates
   - structured LLM call context
   - Polly TTS generation only when a reply is produced
9. Listen for the reply in the voice channel.
10. Send `!leave`.
11. Confirm clean disconnect in logs.

## Rust To Python Request Schema

Each finalized utterance now includes the conversation-aware context Python needs to build a prompt:

- `guild_id`
- `guild_name`
- `voice_channel_id`
- `voice_channel_name`
- `speaker_id`
- `discord_user_id`
- `discord_username`
- `discord_display_name`
- `users_in_call`
- `ssrc`
- `speaker_resolution`
- `utterance_id`
- `sample_rate`
- `channels`
- `audio_base64`

`users_in_call` contains user descriptors with:

- `discord_user_id`
- `username`
- `display_name`

## Prototype Behavior

Example flow:

1. User sends `!join`
2. Bot joins the voice channel
3. User says `butler, what time is it`
4. Rust buffers a short utterance, resolves the speaker from SSRC, and posts speaker-aware metadata plus current participants to Python
5. Python transcribes the utterance and updates per-guild conversation memory
6. Python builds a structured prompt from personality rules, environment context, participants, recent history, and the latest utterance
7. The OpenAI-backed LLM client generates a reply
8. Python generates MP3 TTS for that reply
9. Rust queues the MP3 into the Songbird call only when `should_respond = true`
10. Bot speaks the reply
11. User sends `!leave`

Because memory is speaker-attributed, follow-up turns from different people can stay grounded, for example:

- `[12:01:02] Alice: what time is it`
- `[12:01:05] Butler: It's about 12:01 PM.`
- `[12:01:09] Bob: what about tomorrow`
- `[12:01:13] Butler: Tomorrow is ...`

## Logging and Debugging

Rust logs include:

- bot connected
- joined channel
- speaking SSRC mapped to user
- speaker registry updates
- receiving audio frames
- speech chunk finalized
- transcript text with Discord speaker plus context metadata
- addressed / follow-up / ignored decision
- reply text
- TTS audio queued only when a reply is produced
- disconnect / shutdown

Python logs include:

- health checks
- chunk receive size plus guild/channel metadata
- transcript text with speaker metadata
- addressed vs. ignored decision
- OpenAI request attempts and failures
- chosen reply text or fallback behavior
- Polly TTS generation complete only when a reply is produced

To increase Rust log detail:

```powershell
$env:RUST_LOG="rust_bot=debug,info"
cargo run --manifest-path rust-bot\Cargo.toml
```

## Known Limitations

- TTS uses AWS Polly, so TTS is not fully offline even though STT is local.
- The first `faster-whisper` run downloads the configured model.
- LLM replies require OpenAI API availability and credentials.
- Utterance segmentation is intentionally simple and may miss very quiet speakers.
- Speaker identity is based on Discord/Songbird SSRC mapping, not biometric voice matching.
- Conversation memory is in-memory only and resets when the Python service restarts.
- There is no summarization of older history yet; the service keeps a bounded rolling window only.
- The bot is driven by text commands, not slash commands.
- Multiple people speaking at exactly the same time can still produce imperfect chunking.

## Rough Edges Worth Knowing

- This is a prototype, so short temp audio files are written to `.runtime/`.
- Speaker handling is keyed by SSRC, mapped back to Discord users through `SpeakingStateUpdate`, and persisted as a lightweight per-guild registry.
- The receive path is designed to stay light by doing only chunk buffering in the Songbird event handler and offloading HTTP, LLM, and TTS work to spawned tasks.
- The current LLM client is an OpenAI-backed adapter. A local model can be added later behind the same Python-side interface.

## Source Files

Main Rust bot entrypoint: [rust-bot/src/main.rs](/mnt/c/Users/aljac/Desktop/Butler Discord Frontend/Discord-Voice-Frontend/rust-bot/src/main.rs)

Rust audio helper code: [rust-bot/src/audio.rs](/mnt/c/Users/aljac/Desktop/Butler Discord Frontend/Discord-Voice-Frontend/rust-bot/src/audio.rs)

Rust Python client: [rust-bot/src/python_client.rs](/mnt/c/Users/aljac/Desktop/Butler Discord Frontend/Discord-Voice-Frontend/rust-bot/src/python_client.rs)

Python FastAPI service: [python-service/app.py](/mnt/c/Users/aljac/Desktop/Butler Discord Frontend/Discord-Voice-Frontend/python-service/app.py)

Prompt config: [python-service/prompt_config.py](/mnt/c/Users/aljac/Desktop/Butler Discord Frontend/Discord-Voice-Frontend/python-service/prompt_config.py)
