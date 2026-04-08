use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone)]
pub struct PythonClient {
    base_url: String,
    http: Client,
}

#[derive(Debug, Serialize)]
pub struct AudioProcessRequest {
    pub guild_id: u64,
    pub speaker_id: String,
    pub discord_user_id: Option<u64>,
    pub discord_username: Option<String>,
    pub discord_display_name: Option<String>,
    pub ssrc: u32,
    pub speaker_resolution: String,
    pub utterance_id: u64,
    pub sample_rate: u32,
    pub channels: u16,
    pub audio_base64: String,
}

#[derive(Debug, Deserialize)]
pub struct AudioProcessResponse {
    pub transcript: String,
    pub should_respond: bool,
    pub ignore_reason: Option<String>,
    pub reply_text: Option<String>,
    pub tts_audio_base64: Option<String>,
    pub tts_audio_format: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub whisper_model: String,
}

impl PythonClient {
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_secs(90))
            .build()
            .context("failed to create reqwest client")?;

        Ok(Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            http,
        })
    }

    pub async fn health(&self) -> Result<HealthResponse> {
        let response = self
            .http
            .get(format!("{}/health", self.base_url))
            .send()
            .await
            .context("health request failed")?
            .error_for_status()
            .context("health request returned an error")?;

        response
            .json::<HealthResponse>()
            .await
            .context("failed to parse health response")
    }

    pub async fn process_audio(
        &self,
        mut request: AudioProcessRequest,
        wav_bytes: Vec<u8>,
    ) -> Result<AudioProcessResponse> {
        request.audio_base64 = STANDARD.encode(wav_bytes);

        let response = self
            .http
            .post(format!("{}/process-audio", self.base_url))
            .json(&request)
            .send()
            .await
            .context("audio process request failed")?
            .error_for_status()
            .context("audio process request returned an error")?;

        response
            .json::<AudioProcessResponse>()
            .await
            .context("failed to parse audio process response")
    }

    pub fn decode_audio(&self, audio_base64: &str) -> Result<Vec<u8>> {
        STANDARD
            .decode(audio_base64)
            .context("failed to decode tts audio base64")
    }
}
