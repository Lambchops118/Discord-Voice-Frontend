use anyhow::{Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::io::Cursor;

#[derive(Clone, Debug)]
pub struct AudioPipelineConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub energy_threshold: f32,
    pub silence_frames: usize,
    pub min_speech_frames: usize,
    pub max_speech_frames: usize,
}

impl AudioPipelineConfig {
    pub fn from_env() -> Self {
        let sample_rate = 16_000;
        let channels = 1;
        let min_speech_ms = std::env::var("VOICE_MIN_SPEECH_MS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(350);
        let max_speech_seconds = std::env::var("VOICE_MAX_SPEECH_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(8);

        Self {
            sample_rate,
            channels,
            energy_threshold: std::env::var("VOICE_ENERGY_THRESHOLD")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .unwrap_or(225.0),
            silence_frames: std::env::var("VOICE_SILENCE_FRAMES")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(18),
            min_speech_frames: ((min_speech_ms as f32) / 20.0).ceil() as usize,
            max_speech_frames: ((max_speech_seconds as f32) * 1000.0 / 20.0).ceil() as usize,
        }
    }
}

#[derive(Debug, Default)]
pub struct SpeakerState {
    pub buffer: Vec<i16>,
    pub speech_frames: usize,
    pub silence_frames: usize,
    pub seen_audio_for_current_chunk: bool,
}

impl SpeakerState {
    pub fn begin_if_needed(&mut self) {
        self.seen_audio_for_current_chunk = true;
    }

    pub fn reset(&mut self) -> Vec<i16> {
        self.speech_frames = 0;
        self.silence_frames = 0;
        self.seen_audio_for_current_chunk = false;
        std::mem::take(&mut self.buffer)
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.speech_frames = 0;
        self.silence_frames = 0;
        self.seen_audio_for_current_chunk = false;
    }
}

pub fn frame_energy(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let sum = samples
        .iter()
        .map(|sample| {
            let value = *sample as f32;
            value * value
        })
        .sum::<f32>();

    (sum / samples.len() as f32).sqrt()
}

pub fn write_wav_bytes(samples: &[i16], sample_rate: u32, channels: u16) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut cursor = Cursor::new(Vec::new());
    let mut writer =
        WavWriter::new(&mut cursor, spec).context("failed to create wav writer")?;

    for sample in samples {
        writer
            .write_sample(*sample)
            .context("failed to write wav sample")?;
    }

    writer.finalize().context("failed to finalize wav bytes")?;
    Ok(cursor.into_inner())
}
