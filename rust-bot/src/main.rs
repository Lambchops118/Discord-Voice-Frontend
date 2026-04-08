mod audio;
mod python_client;

use crate::audio::{frame_energy, write_wav_bytes, AudioPipelineConfig, SpeakerState};
use crate::python_client::PythonClient;
use anyhow::{Context as _, Result};
use dashmap::DashMap;
use serenity::{
    async_trait,
    client::{Client, Context, EventHandler as SerenityEventHandler},
    model::{
        channel::Message,
        gateway::Ready,
        id::{ChannelId, GuildId},
    },
};
use songbird::{
    driver::{Channels, DecodeMode, SampleRate},
    input::File as SongbirdFile,
    model::payload::Speaking,
    Event, EventContext as VoiceEventContext, EventHandler as VoiceEventHandler, SerenityInit,
    Songbird,
};
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::fs;
use tracing::{debug, error, info, warn};

#[derive(Clone)]
struct BotState {
    command_prefix: String,
    playback_root: PathBuf,
    audio_config: AudioPipelineConfig,
    python: PythonClient,
    songbird: Arc<Songbird>,
    sessions: Arc<DashMap<GuildId, Arc<GuildAudioSession>>>,
}

struct GuildAudioSession {
    guild_id: GuildId,
    state: BotState,
    speakers: Mutex<std::collections::HashMap<u32, SpeakerState>>,
    utterance_counter: AtomicU64,
}

#[derive(Clone)]
struct BotHandler {
    state: BotState,
}

#[derive(Clone)]
struct VoiceReceiver {
    session: Arc<GuildAudioSession>,
}

struct FinalizedUtterance {
    utterance_id: u64,
    samples: Vec<i16>,
}

#[async_trait]
impl SerenityEventHandler for BotHandler {
    async fn ready(&self, _ctx: Context, ready: Ready) {
        info!("bot connected as {}", ready.user.name);
    }

    async fn message(&self, ctx: Context, msg: Message) {
        if msg.author.bot {
            return;
        }

        let content = msg.content.trim();
        if !content.starts_with(&self.state.command_prefix) {
            return;
        }

        let command = content
            .strip_prefix(&self.state.command_prefix)
            .unwrap_or(content)
            .trim();

        let result = match command {
            "join" => self.handle_join(&ctx, &msg).await,
            "leave" => self.handle_leave(&ctx, &msg).await,
            "status" => self.handle_status(&ctx, &msg).await,
            "pingvoice" => self.handle_pingvoice(&ctx, &msg).await,
            _ => Ok(()),
        };

        if let Err(error) = result {
            error!(?error, "command failed");
            let _ = msg
                .reply(&ctx.http, format!("Voice command failed: {error:#}"))
                .await;
        }
    }
}

#[async_trait]
impl VoiceEventHandler for VoiceReceiver {
    async fn act(&self, ctx: &VoiceEventContext<'_>) -> Option<Event> {
        match ctx {
            VoiceEventContext::SpeakingStateUpdate(speaking) => {
                self.session.on_speaking_update(speaking).await;
            }
            VoiceEventContext::VoiceTick(tick) => {
                self.session.on_voice_tick(tick).await;
            }
            VoiceEventContext::DriverConnect(_data) => {
                info!(guild_id = self.session.guild_id.get(), "voice driver connected");
            }
            VoiceEventContext::DriverReconnect(_data) => {
                warn!(guild_id = self.session.guild_id.get(), "voice driver reconnected");
            }
            VoiceEventContext::DriverDisconnect(data) => {
                warn!(
                    guild_id = self.session.guild_id.get(),
                    disconnect = ?data,
                    "voice driver disconnected"
                );
            }
            VoiceEventContext::ClientDisconnect(data) => {
                info!(
                    guild_id = self.session.guild_id.get(),
                    user_id = ?data.user_id,
                    "client disconnected from voice call"
                );
            }
            _ => {}
        }

        None
    }
}

impl BotHandler {
    async fn handle_join(&self, ctx: &Context, msg: &Message) -> Result<()> {
        let guild_id = msg.guild_id.context("join can only be used in a server")?;
        let channel_id = self
            .author_voice_channel(ctx, msg, guild_id)
            .await?
            .context("you must be in a voice channel before using !join")?;

        if let Some(call) = self.state.songbird.get(guild_id) {
            let current_channel = call.lock().await.current_channel();
            if current_channel == Some(channel_id.into()) {
                msg.reply(&ctx.http, "I am already in your voice channel.")
                    .await
                    .context("failed to send already-connected reply")?;
                return Ok(());
            }
        }

        info!(
            guild_id = guild_id.get(),
            channel_id = channel_id.get(),
            "joining voice channel"
        );

        let call = self
            .state
            .songbird
            .join(guild_id, channel_id)
            .await
            .context("failed to join voice channel")?;

        let session = Arc::new(GuildAudioSession::new(guild_id, self.state.clone()));
        let receiver = VoiceReceiver {
            session: session.clone(),
        };

        {
            let mut handler = call.lock().await;
            handler.remove_all_global_events();
            handler.add_global_event(
                Event::Core(songbird::CoreEvent::SpeakingStateUpdate),
                receiver.clone(),
            );
            handler.add_global_event(
                Event::Core(songbird::CoreEvent::VoiceTick),
                receiver.clone(),
            );
            handler.add_global_event(
                Event::Core(songbird::CoreEvent::DriverConnect),
                receiver.clone(),
            );
            handler.add_global_event(
                Event::Core(songbird::CoreEvent::DriverReconnect),
                receiver.clone(),
            );
            handler.add_global_event(
                Event::Core(songbird::CoreEvent::DriverDisconnect),
                receiver.clone(),
            );
            handler.add_global_event(
                Event::Core(songbird::CoreEvent::ClientDisconnect),
                receiver,
            );
        }

        self.state.sessions.insert(guild_id, session);

        msg.reply(
            &ctx.http,
            format!("Joined <#{}>. Voice receive pipeline is active.", channel_id.get()),
        )
        .await
        .context("failed to send join reply")?;

        Ok(())
    }

    async fn handle_leave(&self, ctx: &Context, msg: &Message) -> Result<()> {
        let guild_id = msg.guild_id.context("leave can only be used in a server")?;

        if self.state.songbird.get(guild_id).is_none() {
            msg.reply(&ctx.http, "I am not in a voice channel right now.")
                .await
                .context("failed to send not-connected reply")?;
            return Ok(());
        }

        info!(guild_id = guild_id.get(), "leaving voice channel");
        self.state.sessions.remove(&guild_id);
        self.state
            .songbird
            .remove(guild_id)
            .await
            .context("failed to remove voice call")?;

        msg.reply(&ctx.http, "Left the voice channel.")
            .await
            .context("failed to send leave reply")?;

        Ok(())
    }

    async fn handle_status(&self, ctx: &Context, msg: &Message) -> Result<()> {
        let guild_id = msg.guild_id.context("status can only be used in a server")?;

        let status = if let Some(call) = self.state.songbird.get(guild_id) {
            let current_channel = call.lock().await.current_channel();
            match current_channel {
                Some(channel_id) => format!(
                    "Connected to <#{}>. Python service reachable: {}",
                    channel_id.0.get(),
                    self.state.python.health().await.is_ok()
                ),
                None => "A voice handler exists, but it is not connected.".to_string(),
            }
        } else {
            "Not connected to voice.".to_string()
        };

        msg.reply(&ctx.http, status)
            .await
            .context("failed to send status reply")?;

        Ok(())
    }

    async fn handle_pingvoice(&self, ctx: &Context, msg: &Message) -> Result<()> {
        let health = self.state.python.health().await?;
        msg.reply(
            &ctx.http,
            format!(
                "Python voice service is {} with whisper model `{}`.",
                health.status, health.whisper_model
            ),
        )
        .await
        .context("failed to send pingvoice reply")?;

        Ok(())
    }

    async fn author_voice_channel(
        &self,
        ctx: &Context,
        msg: &Message,
        guild_id: GuildId,
    ) -> Result<Option<ChannelId>> {
        let guild = msg
            .guild(&ctx.cache)
            .context("failed to read guild from cache")?;

        Ok(guild
            .voice_states
            .get(&msg.author.id)
            .and_then(|voice_state| voice_state.channel_id)
            .filter(|_| guild.id == guild_id))
    }
}

impl GuildAudioSession {
    fn new(guild_id: GuildId, state: BotState) -> Self {
        Self {
            guild_id,
            state,
            speakers: Mutex::new(std::collections::HashMap::new()),
            utterance_counter: AtomicU64::new(1),
        }
    }

    async fn on_speaking_update(&self, speaking: &Speaking) {
        if speaking.user_id.is_some() {
            debug!(
                guild_id = self.guild_id.get(),
                ssrc = speaking.ssrc,
                user_id = ?speaking.user_id,
                "mapped speaking SSRC to user"
            );
        }
    }

    async fn on_voice_tick(self: &Arc<Self>, tick: &songbird::events::context_data::VoiceTick) {
        let mut finalized = Vec::new();

        {
            let mut speakers = match self.speakers.lock() {
                Ok(lock) => lock,
                Err(error) => {
                    error!(?error, "speaker state lock poisoned");
                    return;
                }
            };

            for (ssrc, voice_data) in &tick.speaking {
                let Some(decoded) = &voice_data.decoded_voice else {
                    continue;
                };

                let energy = frame_energy(decoded);
                let state = speakers.entry(*ssrc).or_default();
                let above_threshold = energy >= self.state.audio_config.energy_threshold;

                if above_threshold {
                    if !state.seen_audio_for_current_chunk {
                        debug!(
                            guild_id = self.guild_id.get(),
                            ssrc,
                            energy,
                            "receiving audio frames for utterance"
                        );
                    }

                    state.begin_if_needed();
                    state.silence_frames = 0;
                    state.speech_frames += 1;
                    state.buffer.extend_from_slice(decoded);
                } else if state.seen_audio_for_current_chunk {
                    state.silence_frames += 1;
                    state.buffer.extend_from_slice(decoded);
                }

                if state.seen_audio_for_current_chunk
                    && state.speech_frames >= self.state.audio_config.max_speech_frames
                {
                    finalized.push(self.finish_chunk_locked(*ssrc, state, "max duration"));
                }
            }

            for ssrc in &tick.silent {
                if let Some(state) = speakers.get_mut(ssrc) {
                    if state.seen_audio_for_current_chunk {
                        state.silence_frames += 1;
                        if state.silence_frames >= self.state.audio_config.silence_frames {
                            finalized.push(self.finish_chunk_locked(*ssrc, state, "silence"));
                        }
                    }
                }
            }
        }

        for utterance in finalized.into_iter().flatten() {
            let session = Arc::clone(self);
            tokio::spawn(async move {
                if let Err(error) = session.process_utterance(utterance).await {
                    error!(?error, "failed to process utterance");
                }
            });
        }
    }

    fn finish_chunk_locked(
        &self,
        ssrc: u32,
        state: &mut SpeakerState,
        reason: &str,
    ) -> Option<FinalizedUtterance> {
        if state.speech_frames < self.state.audio_config.min_speech_frames {
            debug!(
                guild_id = self.guild_id.get(),
                ssrc,
                reason,
                speech_frames = state.speech_frames,
                "dropping very short speech chunk"
            );
            state.clear();
            return None;
        }

        let utterance_id = self.utterance_counter.fetch_add(1, Ordering::Relaxed);
        let samples = state.reset();

        info!(
            guild_id = self.guild_id.get(),
            utterance_id,
            ssrc,
            sample_count = samples.len(),
            reason,
            "speech chunk finalized"
        );

        Some(FinalizedUtterance { utterance_id, samples })
    }

    async fn process_utterance(self: Arc<Self>, utterance: FinalizedUtterance) -> Result<()> {
        let wav_bytes = write_wav_bytes(
            &utterance.samples,
            self.state.audio_config.sample_rate,
            self.state.audio_config.channels,
        )?;

        let response = self
            .state
            .python
            .process_audio(
                self.guild_id.get(),
                None,
                utterance.utterance_id,
                self.state.audio_config.sample_rate,
                self.state.audio_config.channels,
                wav_bytes,
            )
            .await?;

        info!(
            guild_id = self.guild_id.get(),
            utterance_id = utterance.utterance_id,
            transcript = response.transcript,
            "received transcript from python service"
        );

        let Some(reply_text) = response.reply_text.clone() else {
            debug!(
                guild_id = self.guild_id.get(),
                utterance_id = utterance.utterance_id,
                "python service returned no reply text"
            );
            return Ok(());
        };

        info!(
            guild_id = self.guild_id.get(),
            utterance_id = utterance.utterance_id,
            reply = reply_text,
            "selected response text"
        );

        let Some(audio_base64) = response.tts_audio_base64.as_deref() else {
            return Ok(());
        };

        let tts_bytes = self.state.python.decode_audio(audio_base64)?;
        let extension = response
            .tts_audio_format
            .as_deref()
            .unwrap_or("mp3")
            .trim_start_matches('.');
        let playback_path = self
            .state
            .playback_root
            .join(format!("guild-{}", self.guild_id.get()))
            .join(format!(
                "utterance-{}-reply-{}.{}",
                utterance.utterance_id,
                millis_since_epoch(),
                extension
            ));

        if let Some(parent) = playback_path.parent() {
            fs::create_dir_all(parent)
                .await
                .with_context(|| format!("failed to create playback directory {}", parent.display()))?;
        }

        fs::write(&playback_path, tts_bytes)
            .await
            .with_context(|| format!("failed to write playback file {}", playback_path.display()))?;

        self.queue_tts_file(&playback_path, utterance.utterance_id).await?;
        self.schedule_cleanup(playback_path);

        Ok(())
    }

    async fn queue_tts_file(&self, path: &Path, utterance_id: u64) -> Result<()> {
        let Some(call) = self.state.songbird.get(self.guild_id) else {
            warn!(
                guild_id = self.guild_id.get(),
                utterance_id,
                "skipping TTS playback because no active call exists"
            );
            return Ok(());
        };

        {
            let mut handler = call.lock().await;
            handler
                .enqueue_input(SongbirdFile::new(path.to_path_buf()).into())
                .await;
        }

        info!(
            guild_id = self.guild_id.get(),
            utterance_id,
            path = %path.display(),
            "audio sent to Discord queue"
        );

        Ok(())
    }

    fn schedule_cleanup(&self, path: PathBuf) {
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(60)).await;
            if let Err(error) = fs::remove_file(&path).await {
                debug!(?error, path = %path.display(), "failed to remove temp playback file");
            }
        });
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    init_tracing();

    let discord_token =
        std::env::var("DISCORD_TOKEN").context("DISCORD_TOKEN must be set in the environment")?;
    let python_service_url = std::env::var("PYTHON_SERVICE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());
    let command_prefix = std::env::var("VOICE_BOT_PREFIX").unwrap_or_else(|_| "!".to_string());
    let playback_root = std::env::var("VOICE_TEMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".runtime"));

    fs::create_dir_all(&playback_root)
        .await
        .with_context(|| format!("failed to create runtime directory {}", playback_root.display()))?;

    let audio_config = AudioPipelineConfig::from_env();
    let songbird_config = songbird::Config::default()
        .decode_mode(DecodeMode::Decode)
        .decode_channels(Channels::Mono)
        .decode_sample_rate(SampleRate::Hz16000);
    let songbird = Songbird::serenity_from_config(songbird_config);

    let state = BotState {
        command_prefix,
        playback_root,
        audio_config,
        python: PythonClient::new(python_service_url)?,
        songbird: songbird.clone(),
        sessions: Arc::new(DashMap::new()),
    };

    let intents = serenity::all::GatewayIntents::GUILDS
        | serenity::all::GatewayIntents::GUILD_MESSAGES
        | serenity::all::GatewayIntents::MESSAGE_CONTENT
        | serenity::all::GatewayIntents::GUILD_VOICE_STATES;

    let mut client = Client::builder(&discord_token, intents)
        .event_handler(BotHandler { state: state.clone() })
        .register_songbird_with(songbird)
        .await
        .context("failed to create serenity client")?;

    let shard_manager = client.shard_manager.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            info!("shutdown signal received");
            shard_manager.shutdown_all().await;
        }
    });

    info!("starting Discord voice bot");
    client.start().await.context("discord client exited with error")
}

fn init_tracing() {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .compact()
        .init();
}

fn millis_since_epoch() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
