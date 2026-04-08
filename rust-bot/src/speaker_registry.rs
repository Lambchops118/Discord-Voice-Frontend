use serenity::model::{
    guild::Member,
    id::{GuildId, UserId},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::fs;
use tracing::{debug, warn};

#[derive(Clone)]
pub struct SpeakerRegistry {
    guild_id: GuildId,
    persistence_path: PathBuf,
    state: Arc<Mutex<SpeakerRegistryState>>,
}

#[derive(Debug, Default)]
struct SpeakerRegistryState {
    active_ssrc_map: HashMap<u32, u64>,
    in_channel_users: HashSet<u64>,
    profiles: HashMap<u64, DiscordSpeakerProfile>,
    speaker_log: HashMap<u64, VoiceIdentityRecord>,
}

#[derive(Clone, Debug)]
pub struct DiscordSpeakerProfile {
    pub discord_user_id: u64,
    pub username: String,
    pub display_name: String,
    pub is_bot: bool,
}

#[derive(Clone, Debug)]
pub struct ResolvedSpeaker {
    pub speaker_id: String,
    pub discord_user_id: Option<u64>,
    pub discord_username: Option<String>,
    pub discord_display_name: Option<String>,
    pub ssrc: u32,
    pub resolved_via: &'static str,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoiceIdentityRecord {
    pub guild_id: u64,
    pub discord_user_id: u64,
    pub username: String,
    pub display_name: String,
    pub latest_ssrc: Option<u32>,
    pub ssrc_history: Vec<u32>,
    pub first_seen_unix_ms: u128,
    pub last_seen_unix_ms: u128,
}

#[derive(Debug, Serialize, Deserialize)]
struct SpeakerRegistrySnapshot {
    guild_id: u64,
    speakers: Vec<VoiceIdentityRecord>,
}

impl DiscordSpeakerProfile {
    pub fn from_member(member: &Member) -> Self {
        Self {
            discord_user_id: member.user.id.get(),
            username: member.user.name.clone(),
            display_name: member.display_name().to_string(),
            is_bot: member.user.bot,
        }
    }
}

impl SpeakerRegistry {
    pub fn new(guild_id: GuildId, persistence_path: PathBuf) -> Self {
        Self {
            guild_id,
            persistence_path,
            state: Arc::new(Mutex::new(SpeakerRegistryState::default())),
        }
    }

    pub fn update_voice_state(
        &self,
        user_id: UserId,
        is_in_session_channel: bool,
        profile: Option<DiscordSpeakerProfile>,
    ) {
        let mut state = match self.state.lock() {
            Ok(lock) => lock,
            Err(error) => {
                warn!(?error, guild_id = self.guild_id.get(), "speaker registry lock poisoned");
                return;
            }
        };

        if let Some(profile) = profile {
            state.profiles.insert(profile.discord_user_id, profile);
        }

        if is_in_session_channel {
            state.in_channel_users.insert(user_id.get());
        } else {
            state.in_channel_users.remove(&user_id.get());
            state.active_ssrc_map.retain(|_, mapped_user_id| *mapped_user_id != user_id.get());
        }
    }

    pub fn update_ssrc_mapping(&self, ssrc: u32, user_id: UserId) {
        let mut state = match self.state.lock() {
            Ok(lock) => lock,
            Err(error) => {
                warn!(?error, guild_id = self.guild_id.get(), "speaker registry lock poisoned");
                return;
            }
        };

        state.active_ssrc_map.insert(ssrc, user_id.get());
        Self::touch_record_locked(&mut state, self.guild_id, user_id.get(), Some(ssrc));
    }

    pub fn resolve_speaker(&self, ssrc: u32) -> ResolvedSpeaker {
        let state = match self.state.lock() {
            Ok(lock) => lock,
            Err(error) => {
                warn!(?error, guild_id = self.guild_id.get(), "speaker registry lock poisoned");
                return Self::unknown_speaker(ssrc);
            }
        };

        let Some(discord_user_id) = state.active_ssrc_map.get(&ssrc).copied() else {
            return Self::resolve_single_participant_fallback(&state, ssrc);
        };

        if let Some(profile) = state.profiles.get(&discord_user_id) {
            return ResolvedSpeaker {
                speaker_id: format!("discord:{discord_user_id}"),
                discord_user_id: Some(discord_user_id),
                discord_username: Some(profile.username.clone()),
                discord_display_name: Some(profile.display_name.clone()),
                ssrc,
                resolved_via: "ssrc_map",
            };
        }

        if let Some(record) = state.speaker_log.get(&discord_user_id) {
            return ResolvedSpeaker {
                speaker_id: format!("discord:{discord_user_id}"),
                discord_user_id: Some(discord_user_id),
                discord_username: Some(record.username.clone()),
                discord_display_name: Some(record.display_name.clone()),
                ssrc,
                resolved_via: "speaker_log",
            };
        }

        ResolvedSpeaker {
            speaker_id: format!("discord:{discord_user_id}"),
            discord_user_id: Some(discord_user_id),
            discord_username: None,
            discord_display_name: Some(format!("discord:{discord_user_id}")),
            ssrc,
            resolved_via: "ssrc_map_user_only",
        }
    }

    fn resolve_single_participant_fallback(
        state: &SpeakerRegistryState,
        ssrc: u32,
    ) -> ResolvedSpeaker {
        let mut candidates = state
            .in_channel_users
            .iter()
            .filter_map(|user_id| state.profiles.get(user_id))
            .filter(|profile| !profile.is_bot)
            .collect::<Vec<_>>();

        if candidates.len() == 1 {
            let profile = candidates.remove(0);
            return ResolvedSpeaker {
                speaker_id: format!("discord:{}", profile.discord_user_id),
                discord_user_id: Some(profile.discord_user_id),
                discord_username: Some(profile.username.clone()),
                discord_display_name: Some(profile.display_name.clone()),
                ssrc,
                resolved_via: "single_channel_member",
            };
        }

        Self::unknown_speaker(ssrc)
    }

    pub fn record_utterance(&self, speaker: &ResolvedSpeaker) {
        let Some(discord_user_id) = speaker.discord_user_id else {
            return;
        };

        let mut state = match self.state.lock() {
            Ok(lock) => lock,
            Err(error) => {
                warn!(?error, guild_id = self.guild_id.get(), "speaker registry lock poisoned");
                return;
            }
        };

        let record = Self::touch_record_locked(
            &mut state,
            self.guild_id,
            discord_user_id,
            Some(speaker.ssrc),
        );

        if let Some(username) = &speaker.discord_username {
            record.username = username.clone();
        }
        if let Some(display_name) = &speaker.discord_display_name {
            record.display_name = display_name.clone();
        }
    }

    pub fn persist_async(&self) {
        let snapshot = match self.snapshot_json() {
            Ok(snapshot) => snapshot,
            Err(error) => {
                warn!(
                    ?error,
                    guild_id = self.guild_id.get(),
                    "failed to serialize speaker registry"
                );
                return;
            }
        };

        let path = self.persistence_path.clone();
        tokio::spawn(async move {
            if let Some(parent) = path.parent() {
                if let Err(error) = fs::create_dir_all(parent).await {
                    warn!(
                        ?error,
                        path = %parent.display(),
                        "failed to create speaker registry directory"
                    );
                    return;
                }
            }

            if let Err(error) = fs::write(&path, snapshot).await {
                warn!(
                    ?error,
                    path = %path.display(),
                    "failed to persist speaker registry"
                );
            } else {
                debug!(path = %path.display(), "persisted speaker registry");
            }
        });
    }

    pub fn current_participants(&self) -> Vec<DiscordSpeakerProfile> {
        let state = match self.state.lock() {
            Ok(lock) => lock,
            Err(error) => {
                warn!(?error, guild_id = self.guild_id.get(), "speaker registry lock poisoned");
                return Vec::new();
            }
        };

        let mut participants = state
            .in_channel_users
            .iter()
            .filter_map(|user_id| state.profiles.get(user_id).cloned())
            .filter(|profile| !profile.is_bot)
            .collect::<Vec<_>>();
        participants.sort_by(|left, right| left.display_name.cmp(&right.display_name));
        participants
    }

    fn snapshot_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        let state = match self.state.lock() {
            Ok(lock) => lock,
            Err(error) => {
                warn!(?error, guild_id = self.guild_id.get(), "speaker registry lock poisoned");
                return serde_json::to_vec_pretty(&SpeakerRegistrySnapshot {
                    guild_id: self.guild_id.get(),
                    speakers: Vec::new(),
                });
            }
        };

        let mut speakers = state.speaker_log.values().cloned().collect::<Vec<_>>();
        speakers.sort_by_key(|speaker| speaker.discord_user_id);

        serde_json::to_vec_pretty(&SpeakerRegistrySnapshot {
            guild_id: self.guild_id.get(),
            speakers,
        })
    }

    fn unknown_speaker(ssrc: u32) -> ResolvedSpeaker {
        let placeholder = format!("unknown:{ssrc}");
        ResolvedSpeaker {
            speaker_id: placeholder.clone(),
            discord_user_id: None,
            discord_username: None,
            discord_display_name: Some(placeholder),
            ssrc,
            resolved_via: "fallback_unknown",
        }
    }

    fn touch_record_locked<'a>(
        state: &'a mut SpeakerRegistryState,
        guild_id: GuildId,
        discord_user_id: u64,
        ssrc: Option<u32>,
    ) -> &'a mut VoiceIdentityRecord {
        let now = unix_millis_now();
        let default_username = format!("discord:{discord_user_id}");
        let default_display_name = default_username.clone();
        let profile = state.profiles.get(&discord_user_id).cloned();
        let record = state
            .speaker_log
            .entry(discord_user_id)
            .or_insert_with(|| VoiceIdentityRecord {
                guild_id: guild_id.get(),
                discord_user_id,
                username: default_username,
                display_name: default_display_name,
                latest_ssrc: None,
                ssrc_history: Vec::new(),
                first_seen_unix_ms: now,
                last_seen_unix_ms: now,
            });

        if let Some(profile) = profile {
            record.username = profile.username;
            record.display_name = profile.display_name;
        }

        record.last_seen_unix_ms = now;

        if let Some(ssrc) = ssrc {
            record.latest_ssrc = Some(ssrc);
            if !record.ssrc_history.contains(&ssrc) {
                record.ssrc_history.push(ssrc);
            }
        }

        record
    }
}

fn unix_millis_now() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
