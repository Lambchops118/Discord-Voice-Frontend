use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Clone)]
pub struct PythonClient {
    base_url: String,
    http: Client,
}

#[derive(Debug, Serialize)]
pub struct CallUser {
    pub discord_user_id: Option<u64>,
    pub username: Option<String>,
    pub display_name: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AudioProcessRequest {
    pub guild_id: u64,
    pub guild_name: Option<String>,
    pub voice_channel_id: Option<u64>,
    pub voice_channel_name: Option<String>,
    pub speaker_id: String,
    pub discord_user_id: Option<u64>,
    pub discord_username: Option<String>,
    pub discord_display_name: Option<String>,
    pub users_in_call: Vec<CallUser>,
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
    pub tts_sample_rate: Option<u32>,
    pub tts_channels: Option<u16>,
}

/// One decoded frame of the `/process-audio/stream` NDJSON response.
#[derive(Debug)]
pub enum StreamFrame {
    /// Sent first: transcript + whether a reply is coming.
    Meta {
        transcript: String,
        should_respond: bool,
        ignore_reason: Option<String>,
    },
    /// One synthesized sentence, ready to enqueue for playback.
    Audio {
        seq: u64,
        text: String,
        audio_format: String,
        sample_rate: u32,
        channels: u16,
        pcm: Vec<u8>,
    },
    /// Sent last on success: the full reply text and chunk count.
    End {
        reply_text: String,
        chunks: u64,
    },
    /// Server-side error mid-stream.
    Error { message: String },
}

/// Wire form of a single NDJSON line; `type` selects the variant.
#[derive(Debug, Deserialize)]
struct RawStreamFrame {
    #[serde(rename = "type")]
    kind: String,
    transcript: Option<String>,
    should_respond: Option<bool>,
    ignore_reason: Option<String>,
    seq: Option<u64>,
    text: Option<String>,
    audio_format: Option<String>,
    sample_rate: Option<u32>,
    channels: Option<u16>,
    pcm_base64: Option<String>,
    reply_text: Option<String>,
    chunks: Option<u64>,
    message: Option<String>,
}

impl RawStreamFrame {
    fn into_frame(self) -> Result<StreamFrame> {
        match self.kind.as_str() {
            "meta" => Ok(StreamFrame::Meta {
                transcript: self.transcript.unwrap_or_default(),
                should_respond: self.should_respond.unwrap_or(false),
                ignore_reason: self.ignore_reason,
            }),
            "audio" => {
                let pcm = STANDARD
                    .decode(self.pcm_base64.unwrap_or_default())
                    .context("failed to decode streamed pcm base64")?;
                Ok(StreamFrame::Audio {
                    seq: self.seq.unwrap_or(0),
                    text: self.text.unwrap_or_default(),
                    audio_format: self.audio_format.unwrap_or_else(|| "pcm".to_string()),
                    sample_rate: self.sample_rate.unwrap_or(16_000),
                    channels: self.channels.unwrap_or(1),
                    pcm,
                })
            }
            "end" => Ok(StreamFrame::End {
                reply_text: self.reply_text.unwrap_or_default(),
                chunks: self.chunks.unwrap_or(0),
            }),
            "error" => Ok(StreamFrame::Error {
                message: self.message.unwrap_or_else(|| "unknown error".to_string()),
            }),
            other => Err(anyhow!("unknown stream frame type: {other}")),
        }
    }
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

    /// Open the streaming pipeline: POST the utterance and receive decoded
    /// [`StreamFrame`]s over a channel as the reply is generated and synthesized.
    ///
    /// A background task reframes the chunked NDJSON body (frames can straddle
    /// HTTP chunk boundaries) and forwards each parsed frame in order, so the
    /// caller can enqueue audio for playback the moment each sentence lands.
    pub async fn process_audio_stream(
        &self,
        mut request: AudioProcessRequest,
        wav_bytes: Vec<u8>,
    ) -> Result<mpsc::Receiver<Result<StreamFrame>>> {
        request.audio_base64 = STANDARD.encode(wav_bytes);

        let response = self
            .http
            .post(format!("{}/process-audio/stream", self.base_url))
            .json(&request)
            .send()
            .await
            .context("audio stream request failed")?
            .error_for_status()
            .context("audio stream request returned an error")?;

        let (tx, rx) = mpsc::channel::<Result<StreamFrame>>(32);
        tokio::spawn(async move {
            let mut stream = response.bytes_stream();
            let mut buffer: Vec<u8> = Vec::new();

            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        let _ = tx
                            .send(Err(anyhow!("stream body read failed: {error}")))
                            .await;
                        return;
                    }
                };
                buffer.extend_from_slice(&chunk);

                // Emit every complete newline-delimited frame in the buffer.
                while let Some(pos) = buffer.iter().position(|byte| *byte == b'\n') {
                    let line: Vec<u8> = buffer.drain(..=pos).collect();
                    if !forward_line(&tx, &line[..line.len() - 1]).await {
                        return;
                    }
                }
            }

            // Flush any trailing frame that arrived without a final newline.
            let _ = forward_line(&tx, &buffer).await;
        });

        Ok(rx)
    }

    pub fn decode_audio(&self, audio_base64: &str) -> Result<Vec<u8>> {
        STANDARD
            .decode(audio_base64)
            .context("failed to decode tts audio base64")
    }
}

/// Parse one NDJSON line and forward it. Returns `false` if the receiver is gone
/// (caller should stop) or the line was an unrecoverable parse error.
async fn forward_line(tx: &mpsc::Sender<Result<StreamFrame>>, line: &[u8]) -> bool {
    let trimmed = line.strip_suffix(b"\r").unwrap_or(line);
    if trimmed.is_empty() {
        return true;
    }
    let parsed = serde_json::from_slice::<RawStreamFrame>(trimmed)
        .context("failed to parse stream frame json")
        .and_then(RawStreamFrame::into_frame);
    tx.send(parsed).await.is_ok()
}
