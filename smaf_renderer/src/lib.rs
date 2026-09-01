use std::sync::Arc;

use smaf_player::{MaFmNote, MaFmOperatorPatch, MaFmPatch, SmafEvent, WaveDynamics};

const TWO_PI: f64 = core::f64::consts::PI * 2.0;
// Tuned to the MA/YMF family rather than a generic full-depth phase modulator.
// A full 1.0-cycle modulation index is the main cause of the characteristic
// "screaming" approximation heard in older SMAF players.
const MODULATION_DEPTH_CYCLES: f64 = 0.42;
const MAX_RELEASE_MS: u32 = 5_000;
const FM_NOTE_HEADROOM: f64 = 0.224;
const OUTPUT_LOWPASS_HZ: f64 = 10_000.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum EnvelopeStage {
    Attack,
    Decay,
    Sustain,
    Release,
    Done,
}

#[derive(Clone, Copy)]
struct Envelope {
    stage: EnvelopeStage,
    level: f64,
    attack_step: f64,
    decay_step: f64,
    sustain_step: f64,
    release_step: f64,
    sustain_level: f64,
    ignore_key_off: bool,
}

impl Envelope {
    fn new(patch: &MaFmOperatorPatch, sample_rate: f64, rate_scale: f64) -> Self {
        Self {
            stage: EnvelopeStage::Attack,
            level: 0.0,
            attack_step: envelope_step(patch.ar, sample_rate, true) * rate_scale,
            decay_step: envelope_step(patch.dr, sample_rate, false) * rate_scale,
            sustain_step: envelope_step(patch.sr, sample_rate, false) * rate_scale,
            release_step: envelope_step(patch.rr, sample_rate, false) * rate_scale,
            // YMF825/VM35 SL is inverse: 0 = no sustain attenuation,
            // 15 = silent. The exact hardware curve is not public here, so keep
            // the monotonic field semantics without inventing a 45 dB range.
            sustain_level: 1.0 - f64::from(patch.sl.min(15)) / 15.0,
            ignore_key_off: patch.xof,
        }
    }

    fn key_off(&mut self) {
        // XOF explicitly means Ignore KeyOff in VM35. Such an operator keeps
        // following its own decay/sustain envelope. The renderer has a finite
        // voice-tail ceiling so malformed or intentionally infinite patches do
        // not keep an emulator mixer alive forever.
        if self.ignore_key_off || self.stage == EnvelopeStage::Done {
            return;
        }
        self.stage = EnvelopeStage::Release;
    }

    fn tick(&mut self) -> f64 {
        match self.stage {
            EnvelopeStage::Attack => {
                // Hardware semantics: AR=0 never rises. This is distinct from
                // DR/SR/RR=0, which freeze their respective attenuation stage.
                if self.attack_step <= 0.0 {
                    return 0.0;
                }
                self.level += self.attack_step;
                if self.level >= 1.0 {
                    self.level = 1.0;
                    self.stage = EnvelopeStage::Decay;
                }
            }
            EnvelopeStage::Decay => {
                if self.decay_step > 0.0 {
                    self.level -= self.decay_step;
                    if self.level <= self.sustain_level {
                        self.level = self.sustain_level;
                        self.stage = EnvelopeStage::Sustain;
                    }
                }
            }
            EnvelopeStage::Sustain => {
                if self.sustain_step > 0.0 {
                    self.level -= self.sustain_step;
                    if self.level <= 0.0 {
                        self.level = 0.0;
                        self.stage = EnvelopeStage::Done;
                    }
                }
            }
            EnvelopeStage::Release => {
                if self.release_step > 0.0 {
                    self.level -= self.release_step;
                    if self.level <= 0.0 {
                        self.level = 0.0;
                        self.stage = EnvelopeStage::Done;
                    }
                }
            }
            EnvelopeStage::Done => return 0.0,
        }
        self.level.clamp(0.0, 1.0)
    }

    fn finished(self) -> bool {
        self.stage == EnvelopeStage::Done
    }
}

#[derive(Clone, Copy)]
struct Operator {
    patch: MaFmOperatorPatch,
    envelope: Envelope,
    phase: f64,
    phase_step: f64,
    level_gain: f64,
    feedback: [f64; 2],
    muted_by_nyquist: bool,
}

impl Operator {
    fn new(patch: MaFmOperatorPatch, base_frequency: f64, midi_note: f64, sample_rate: f64) -> Self {
        // MA/YMF825 MULTI is linear: 0 means x0.5, 1..15 mean x1..x15.
        let multiple = if patch.multi == 0 { 0.5 } else { f64::from(patch.multi & 15) };
        let detune_cents = (f64::from(patch.dt) - 3.5) * 1.2;
        let detune = 2.0_f64.powf(detune_cents / 1200.0);
        let operator_frequency = base_frequency * multiple * detune;
        let phase_step = operator_frequency / sample_rate;

        let octaves_above_middle_c = ((midi_note - 60.0) / 12.0).max(0.0);
        let ksl_db_per_octave = [0.0, 1.5, 3.0, 6.0][usize::from(patch.ksl & 3)];
        let level_gain = db_to_gain(-0.75 * f64::from(patch.tl)) * db_to_gain(-ksl_db_per_octave * octaves_above_middle_c);

        let semitones_above_middle_c = midi_note - 60.0;
        let rate_scale = 2.0_f64
            .powf(semitones_above_middle_c / if patch.ksr { 24.0 } else { 96.0 })
            .clamp(0.5, 4.0);

        Self {
            patch,
            envelope: Envelope::new(&patch, sample_rate, rate_scale),
            phase: 0.0,
            phase_step,
            level_gain,
            feedback: [0.0; 2],
            muted_by_nyquist: phase_step >= 0.5,
        }
    }

    fn key_off(&mut self) {
        self.envelope.key_off();
    }

    fn tick(&mut self, external_modulation_cycles: f64, lfo: f64) -> f64 {
        let envelope = self.envelope.tick();
        if self.muted_by_nyquist || envelope <= 0.0 {
            return 0.0;
        }

        let feedback_amount = if self.patch.feedback == 0 {
            0.0
        } else {
            (self.feedback[0] + self.feedback[1]) * 0.5 * f64::from(self.patch.feedback) / 24.0
        };

        let vibrato_depth = [0.00196, 0.00387, 0.00774, 0.01548][usize::from(self.patch.dvb & 3)];
        let step = if self.patch.vib {
            self.phase_step * (1.0 + lfo * vibrato_depth)
        } else {
            self.phase_step
        };
        self.phase = (self.phase + step).fract();

        let tremolo_depth = [0.129, 0.242, 0.424, 0.669][usize::from(self.patch.dam & 3)];
        let tremolo = if self.patch.am { 1.0 - tremolo_depth * (0.5 + 0.5 * lfo) } else { 1.0 };

        let value =
            waveform(self.patch.wave, self.phase + external_modulation_cycles + feedback_amount, step.abs()) * self.level_gain * envelope * tremolo;
        self.feedback[1] = self.feedback[0];
        self.feedback[0] = value;
        value
    }

    fn finished(self) -> bool {
        self.envelope.finished()
    }
}

struct Voice {
    patch: MaFmPatch,
    operators: [Operator; 4],
    lfo_phase: f64,
    lfo_step: f64,
}

impl Voice {
    fn new(patch: MaFmPatch, frequency: f64, midi_note: f64, sample_rate: f64) -> Self {
        let operators = core::array::from_fn(|index| Operator::new(patch.operators[index], frequency, midi_note, sample_rate));
        let lfo_hz = [1.8, 4.0, 6.0, 9.7][usize::from(patch.lfo & 3)];
        Self {
            patch,
            operators,
            lfo_phase: 0.0,
            lfo_step: lfo_hz / sample_rate,
        }
    }

    fn key_off(&mut self) {
        for operator in self.operators.iter_mut().take(self.patch.operator_count as usize) {
            operator.key_off();
        }
    }

    fn tick(&mut self) -> f64 {
        self.lfo_phase = (self.lfo_phase + self.lfo_step).fract();
        let lfo = (TWO_PI * self.lfo_phase).sin();
        let d = MODULATION_DEPTH_CYCLES;
        let op = &mut self.operators;

        match self.patch.algorithm & 7 {
            0 => {
                let a = op[0].tick(0.0, lfo);
                op[1].tick(a * d, lfo)
            }
            1 => (op[0].tick(0.0, lfo) + op[1].tick(0.0, lfo)) * 0.5,
            2 => (op[0].tick(0.0, lfo) + op[1].tick(0.0, lfo) + op[2].tick(0.0, lfo) + op[3].tick(0.0, lfo)) * 0.35,
            3 => {
                let a = op[0].tick(0.0, lfo);
                let b = op[1].tick(0.0, lfo);
                let c = op[2].tick((a + b) * d, lfo);
                op[3].tick(c * d, lfo)
            }
            4 => {
                let a = op[0].tick(0.0, lfo);
                let b = op[1].tick(a * d, lfo);
                let c = op[2].tick(b * d, lfo);
                op[3].tick(c * d, lfo)
            }
            5 => {
                let a = op[0].tick(0.0, lfo);
                let b = op[1].tick(a * d, lfo);
                let c = op[2].tick(0.0, lfo);
                let e = op[3].tick(c * d, lfo);
                (b + e) * 0.5
            }
            6 => {
                let a = op[0].tick(0.0, lfo);
                let b = op[1].tick(0.0, lfo);
                let c = op[2].tick(b * d, lfo);
                let e = op[3].tick(c * d, lfo);
                (a + e) * 0.5
            }
            _ => {
                let a = op[0].tick(0.0, lfo);
                let b = op[1].tick(0.0, lfo);
                let c = op[2].tick(b * d, lfo);
                let e = op[3].tick(0.0, lfo);
                (a + c + e) * 0.4
            }
        }
    }

    fn finished(&self) -> bool {
        // A modulator with RR=0/XOF can legitimately keep running after every
        // audible carrier is silent. Waiting for all operators creates several
        // seconds of false tail. Playback is complete when all carriers for the
        // selected connection algorithm have completed.
        carrier_indices(self.patch.algorithm, self.patch.operator_count)
            .iter()
            .all(|index| self.operators[*index].finished())
    }
}

pub struct RenderedFmNote {
    pub channels: u8,
    pub sampling_rate: u32,
    pub data: Vec<i16>,
}

/// Deterministic stereo PCM for the audio carried by the SMAF file itself.
///
/// This deliberately renders only `Wave` and `MaFmNote`. Generic MIDI events
/// represent MA handset-ROM instruments which are not contained in the MMF and
/// must remain on the host's explicit ROM/SF2 fallback path. Keeping that
/// boundary here prevents a SoundFont from replacing file-authored FM voices or
/// embedded samples such as ZIC2 B7's bird-call Mwa.
pub struct RenderedEmbeddedAudio {
    pub channels: u8,
    pub sampling_rate: u32,
    pub data: Vec<i16>,
}

/// Incremental renderer for the audio authored inside one SMAF sequence.
///
/// It uses the same 24 kHz-style source clock and per-note signal path as
/// [`render_embedded_audio`], but retains oscillator, envelope, resampler and
/// filter state between calls instead of allocating an entire song-sized stem.
pub struct EmbeddedRenderer {
    events: Arc<Vec<(usize, SmafEvent)>>,
    sample_rate: u32,
    frame: usize,
    next_event: usize,
    active: Vec<ActiveEmbeddedVoice>,
}

impl EmbeddedRenderer {
    pub fn new(events: Arc<Vec<(usize, SmafEvent)>>, sample_rate: u32) -> Self {
        Self {
            events,
            sample_rate: sample_rate.max(8_000),
            frame: 0,
            next_event: 0,
            active: Vec::new(),
        }
    }

    /// Starts the event timeline again while allowing release tails from the
    /// previous cycle to finish, matching the former overlapping PCM stems.
    pub fn restart_cycle(&mut self) {
        self.frame = 0;
        self.next_event = 0;
    }

    pub fn next_frame(&mut self) -> Option<[i16; 2]> {
        while let Some((time_ms, event)) = self.events.get(self.next_event) {
            let event_frame = time_ms.saturating_mul(self.sample_rate as usize) / 1000;
            if event_frame > self.frame {
                break;
            }

            match event {
                SmafEvent::Wave {
                    channels,
                    sampling_rate,
                    data,
                    dynamics,
                } => {
                    if let Some(voice) = WaveVoice::new(*channels, *sampling_rate, data.clone(), *dynamics, self.sample_rate) {
                        self.active.push(ActiveEmbeddedVoice::Wave(voice));
                    }
                }
                SmafEvent::MaFmNote(note) => self
                    .active
                    .push(ActiveEmbeddedVoice::Fm(Box::new(FmVoiceRenderer::new(note, self.sample_rate)))),
                SmafEvent::MidiNoteOn { .. }
                | SmafEvent::MidiNoteOff { .. }
                | SmafEvent::MidiProgramChange { .. }
                | SmafEvent::MidiControlChange { .. }
                | SmafEvent::MidiPitchBend { .. }
                | SmafEvent::MidiSysEx(_)
                | SmafEvent::End => {}
            }
            self.next_event += 1;
        }

        if self.next_event == self.events.len() && self.active.is_empty() {
            return None;
        }

        let mut mixed = [0.0; 2];
        let mut produced = false;
        let mut index = 0;
        while index < self.active.len() {
            let (sample, finished) = self.active[index].next_frame();
            if let Some(sample) = sample {
                mixed[0] += sample[0];
                mixed[1] += sample[1];
                produced = true;
            }
            if finished {
                self.active.remove(index);
            } else {
                index += 1;
            }
        }

        if !produced && self.next_event == self.events.len() && self.active.is_empty() {
            return None;
        }

        self.frame = self.frame.saturating_add(1);
        Some([float_to_i16(mixed[0]), float_to_i16(mixed[1])])
    }
}

enum ActiveEmbeddedVoice {
    Wave(WaveVoice),
    Fm(Box<FmVoiceRenderer>),
}

impl ActiveEmbeddedVoice {
    fn next_frame(&mut self) -> (Option<[f64; 2]>, bool) {
        match self {
            Self::Wave(voice) => {
                let (sample, finished) = voice.next_frame();
                (Some(sample), finished)
            }
            Self::Fm(voice) => match voice.next_frame() {
                Some(frame) => (Some([f64::from(frame[0]) / 32768.0, f64::from(frame[1]) / 32768.0]), false),
                None => (None, true),
            },
        }
    }
}

struct WaveVoice {
    channels: usize,
    source_rate: u32,
    source: Vec<i16>,
    target_rate: u32,
    output_frame: usize,
    output_frames: usize,
    gain: f64,
    left_gain: f64,
    right_gain: f64,
}

impl WaveVoice {
    fn new(channels: u8, source_rate: u32, source: Vec<i16>, dynamics: WaveDynamics, target_rate: u32) -> Option<Self> {
        if !matches!(channels, 1 | 2) || source_rate == 0 {
            return None;
        }
        let channels = channels as usize;
        let source_frames = source.len() / channels;
        if source_frames == 0 {
            return None;
        }
        let output_frames =
            ((source_frames as u64 * u64::from(target_rate)).saturating_add(u64::from(source_rate) - 1) / u64::from(source_rate)) as usize;
        let gain = f64::from(dynamics.velocity) / 127.0 * (f64::from(dynamics.volume) / 127.0) * (f64::from(dynamics.expression) / 127.0);
        let pan = dynamics
            .pan
            .map(|value| ((f64::from(value.min(127)) - 64.0) / 63.0).clamp(-1.0, 1.0))
            .unwrap_or(0.0);

        Some(Self {
            channels,
            source_rate,
            source,
            target_rate,
            output_frame: 0,
            output_frames,
            gain,
            left_gain: if pan > 0.0 { 1.0 - pan } else { 1.0 },
            right_gain: if pan < 0.0 { 1.0 + pan } else { 1.0 },
        })
    }

    fn next_frame(&mut self) -> ([f64; 2], bool) {
        if self.output_frame >= self.output_frames {
            return ([0.0; 2], true);
        }

        let source_frames = self.source.len() / self.channels;
        let numerator = self.output_frame as u64 * u64::from(self.source_rate);
        let left_index = (numerator / u64::from(self.target_rate)) as usize;
        let fraction = (numerator % u64::from(self.target_rate)) as f64 / f64::from(self.target_rate);
        let right_index = (left_index + 1).min(source_frames - 1);
        let mut result = [0.0; 2];
        for (target_channel, result) in result.iter_mut().enumerate() {
            let source_channel = if self.channels == 1 { 0 } else { target_channel };
            let left = f64::from(self.source[left_index * self.channels + source_channel]) / 32768.0;
            let right = f64::from(self.source[right_index * self.channels + source_channel]) / 32768.0;
            let pan_gain = if target_channel == 0 { self.left_gain } else { self.right_gain };
            *result = (left + (right - left) * fraction) * self.gain * pan_gain;
        }

        self.output_frame += 1;
        (result, self.output_frame >= self.output_frames)
    }
}

pub fn render_embedded_audio(events: &[(usize, SmafEvent)], sample_rate: u32) -> RenderedEmbeddedAudio {
    let sample_rate = sample_rate.max(8_000);
    let mut renderer = EmbeddedRenderer::new(Arc::new(events.to_vec()), sample_rate);
    let mut data = Vec::new();
    while let Some(frame) = renderer.next_frame() {
        data.extend_from_slice(&frame);
    }
    RenderedEmbeddedAudio {
        channels: 2,
        sampling_rate: sample_rate,
        data,
    }
}

/// Render one decoded SMAF Wave event with its authored note/channel dynamics.
///
/// The waveform decoder remains bit-exact and gain-free; velocity, Volume,
/// Expression and Pan belong to the sequence/renderer layer. This is the same
/// path used by `render_embedded_audio` and is suitable for a host emulator's
/// per-event audio backend.
pub fn render_wave_event(channels: u8, source_rate: u32, source: &[i16], dynamics: WaveDynamics, target_rate: u32) -> RenderedEmbeddedAudio {
    let target_rate = target_rate.max(1);
    let mut mix = Vec::new();
    mix_wave(&mut mix, 0, target_rate, channels, source_rate, source, dynamics);
    RenderedEmbeddedAudio {
        channels: 2,
        sampling_rate: target_rate,
        data: mix.into_iter().map(float_to_i16).collect(),
    }
}

fn mix_wave(mix: &mut Vec<f64>, start_frame: usize, target_rate: u32, channels: u8, source_rate: u32, source: &[i16], dynamics: WaveDynamics) {
    if !matches!(channels, 1 | 2) || source_rate == 0 {
        return;
    }
    let source_channels = channels as usize;
    let source_frames = source.len() / source_channels;
    if source_frames == 0 {
        return;
    }
    let output_frames =
        ((source_frames as u64 * u64::from(target_rate)).saturating_add(u64::from(source_rate) - 1) / u64::from(source_rate)) as usize;
    let needed = start_frame.saturating_add(output_frames).saturating_mul(2);
    if mix.len() < needed {
        mix.resize(needed, 0.0);
    }

    let gain = f64::from(dynamics.velocity) / 127.0 * (f64::from(dynamics.volume) / 127.0) * (f64::from(dynamics.expression) / 127.0);
    let pan = dynamics
        .pan
        .map(|value| ((f64::from(value.min(127)) - 64.0) / 63.0).clamp(-1.0, 1.0))
        .unwrap_or(0.0);
    let left_gain = if pan > 0.0 { 1.0 - pan } else { 1.0 };
    let right_gain = if pan < 0.0 { 1.0 + pan } else { 1.0 };

    for output_frame in 0..output_frames {
        let numerator = output_frame as u64 * u64::from(source_rate);
        let left_index = (numerator / u64::from(target_rate)) as usize;
        let fraction = (numerator % u64::from(target_rate)) as f64 / f64::from(target_rate);
        let right_index = (left_index + 1).min(source_frames - 1);

        for target_channel in 0..2usize {
            let source_channel = if source_channels == 1 { 0 } else { target_channel };
            let left = f64::from(source[left_index * source_channels + source_channel]) / 32768.0;
            let right = f64::from(source[right_index * source_channels + source_channel]) / 32768.0;
            let sample = left + (right - left) * fraction;
            let pan_gain = if target_channel == 0 { left_gain } else { right_gain };
            mix[(start_frame + output_frame) * 2 + target_channel] += sample * gain * pan_gain;
        }
    }
}

pub fn render_ma_fm_note(note: &MaFmNote, sample_rate: u32) -> RenderedFmNote {
    let sample_rate = sample_rate.max(8_000);
    let mut renderer = FmVoiceRenderer::new(note, sample_rate);
    let mut data = Vec::new();
    while let Some(frame) = renderer.next_frame() {
        data.extend_from_slice(&frame);
    }

    RenderedFmNote {
        channels: 2,
        sampling_rate: sample_rate,
        data,
    }
}

struct FmVoiceRenderer {
    voice: Voice,
    gate_frames_remaining: usize,
    release_frames_rendered: usize,
    release_limit: usize,
    gain: f64,
    left_gain: f64,
    right_gain: f64,
    lowpass: TwoPoleLowpass,
    key_off: bool,
}

impl FmVoiceRenderer {
    fn new(note: &MaFmNote, sample_rate: u32) -> Self {
        let bend = (f64::from(note.pitch_bend.min(0x3fff)) - 8192.0) / 8192.0 * 2.0;
        let midi_note = f64::from(note.note) + f64::from(note.patch.note_shift) + bend;
        let frequency = 440.0 * 2.0_f64.powf((midi_note - 69.0) / 12.0);
        let voice = Voice::new(note.patch, frequency, midi_note, f64::from(sample_rate));

        let velocity = f64::from(note.velocity) / 127.0;
        let channel_volume = f64::from(note.volume) / 127.0;
        let expression = f64::from(note.expression) / 127.0;
        // Dense MA scores routinely overlap many operators/voices. Keep the same
        // per-note headroom as a handset-style mix bus rather than clipping each
        // note close to full scale before rodio mixes them.
        let gain = velocity * channel_volume * expression * FM_NOTE_HEADROOM;

        let channel_pan = note.pan.map(|value| (f64::from(value) - 64.0) / 63.0).unwrap_or(0.0);
        let pan = (channel_pan + f64::from(note.patch.default_pan)).clamp(-1.0, 1.0);
        let left_gain = if pan > 0.0 { 1.0 - pan } else { 1.0 };
        let right_gain = if pan < 0.0 { 1.0 + pan } else { 1.0 };

        Self {
            voice,
            gate_frames_remaining: (u64::from(note.gate_ms) * u64::from(sample_rate) / 1000) as usize,
            release_frames_rendered: 0,
            release_limit: (u64::from(MAX_RELEASE_MS) * u64::from(sample_rate) / 1000) as usize,
            gain,
            left_gain,
            right_gain,
            lowpass: TwoPoleLowpass::new(sample_rate, OUTPUT_LOWPASS_HZ),
            key_off: false,
        }
    }

    fn next_frame(&mut self) -> Option<[i16; 2]> {
        if self.gate_frames_remaining != 0 {
            self.gate_frames_remaining -= 1;
        } else {
            if !self.key_off {
                self.voice.key_off();
                self.key_off = true;
            }
            if self.release_frames_rendered >= self.release_limit || self.voice.finished() {
                return None;
            }
            self.release_frames_rendered += 1;
        }

        let sample = self.voice.tick() * self.gain;
        let raw = [float_to_i16(sample * self.left_gain), float_to_i16(sample * self.right_gain)];
        Some(self.lowpass.process(raw))
    }
}

struct TwoPoleLowpass {
    coefficient: f64,
    first: [f64; 2],
    second: [f64; 2],
}

impl TwoPoleLowpass {
    fn new(sample_rate: u32, cutoff_hz: f64) -> Self {
        let cutoff_hz = cutoff_hz.min(f64::from(sample_rate) * 0.45).max(20.0);
        Self {
            coefficient: 1.0 - (-TWO_PI * cutoff_hz / f64::from(sample_rate)).exp(),
            first: [0.0; 2],
            second: [0.0; 2],
        }
    }

    fn process(&mut self, mut frame: [i16; 2]) -> [i16; 2] {
        for (channel, sample) in frame.iter_mut().enumerate() {
            let input = f64::from(*sample) / 32768.0;
            self.first[channel] += self.coefficient * (input - self.first[channel]);
            self.second[channel] += self.coefficient * (self.first[channel] - self.second[channel]);
            *sample = float_to_i16(self.second[channel]);
        }
        frame
    }
}

fn float_to_i16(value: f64) -> i16 {
    (value.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

fn db_to_gain(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

fn envelope_step(rate: u8, sample_rate: f64, attack: bool) -> f64 {
    let rate = rate.min(15);
    if rate == 0 {
        return 0.0;
    }
    // The MA/YMF rate fields are exponential control values. Public material
    // documents ordering and zero-rate semantics but not a bit-exact time table,
    // so interpolate in log-time between empirically stable endpoints. Keep this
    // isolated so a measured hardware table can replace it without touching the
    // event/parser layers.
    let position = (f64::from(rate) - 1.0) / 14.0;
    let slow_seconds: f64 = if attack { 0.35 } else { 4.0 };
    let fast_seconds: f64 = if attack { 0.0008 } else { 0.004 };
    let seconds = slow_seconds * (fast_seconds / slow_seconds).powf(position);
    1.0 / (seconds * sample_rate).max(1.0)
}

fn carrier_indices(algorithm: u8, operator_count: u8) -> &'static [usize] {
    if operator_count <= 2 {
        return match algorithm & 1 {
            0 => &[1],
            _ => &[0, 1],
        };
    }
    match algorithm & 7 {
        0 => &[1],
        1 => &[0, 1],
        2 => &[0, 1, 2, 3],
        3 | 4 => &[3],
        5 => &[1, 3],
        6 => &[0, 3],
        _ => &[0, 2, 3],
    }
}

fn waveform(wave: u8, phase: f64, phase_step: f64) -> f64 {
    let phase = phase - phase.floor();
    match wave & 31 {
        // WS0..7 are the OPL3-compatible family.
        0..=7 => opl_family_wave(wave & 7, phase, phase_step),

        // WS8..13 are documented as amplitude-clipped variants of WS0..5.
        // The exact clipping threshold is hardware-specific; use a soft clip to
        // preserve the family while avoiding the aliasing of an arbitrary hard
        // threshold.
        8..=13 => soft_clip(opl_family_wave((wave - 8) & 7, phase, phase_step)),

        // WS14/22/30 are square-wave variants. Their exact duty/shape differs on
        // hardware; the band-limited square is a deterministic safe family
        // approximation and is preferable to aliasing them back to sine waves.
        14 | 22 | 30 => band_limited_square(phase, phase_step),

        // WS15/23/31 are waveform-memory selectors on MA-3/MA-5. A VM35 patch
        // alone does not contain that RAM waveform. Silence is safer than
        // fabricating an unrelated oscillator. ZIC2's custom-FM corpus does not
        // use these selectors; the validator enforces that invariant.
        15 | 23 | 31 => 0.0,

        // WS16..21 are triangle-family variants.
        16..=21 => family_variant(band_limited_triangle(phase, phase_step), phase, wave - 16),

        // WS24..29 are saw-family variants.
        24..=29 => family_variant(band_limited_saw(phase, phase_step), phase, wave - 24),
        _ => 0.0,
    }
}

fn opl_family_wave(selector: u8, phase: f64, phase_step: f64) -> f64 {
    let sine = (TWO_PI * phase).sin();
    match selector & 7 {
        0 => sine,
        1 => sine.max(0.0),
        2 => sine.abs(),
        3 => {
            let folded = phase - (phase * 2.0).floor() * 0.5;
            if folded < 0.25 {
                (TWO_PI * folded).sin().abs()
            } else {
                0.0
            }
        }
        4 => {
            if phase < 0.5 {
                (TWO_PI * phase * 2.0).sin()
            } else {
                0.0
            }
        }
        5 => {
            if phase < 0.5 {
                (TWO_PI * phase * 2.0).sin().abs()
            } else {
                0.0
            }
        }
        6 => band_limited_square(phase, phase_step),
        _ => band_limited_saw(phase, phase_step) * 0.8,
    }
}

fn family_variant(base: f64, phase: f64, variant: u8) -> f64 {
    match variant {
        0 => base,
        1 => base.max(0.0),
        2 => base.abs(),
        3 => {
            if !(0.25..0.75).contains(&phase) {
                base.abs()
            } else {
                0.0
            }
        }
        4 => {
            if phase < 0.5 {
                base
            } else {
                0.0
            }
        }
        _ => {
            if phase < 0.5 {
                base.abs()
            } else {
                0.0
            }
        }
    }
}

fn soft_clip(value: f64) -> f64 {
    const DRIVE: f64 = 2.25;
    (value * DRIVE).tanh() / DRIVE.tanh()
}

fn harmonic_limit(phase_step: f64, cap: usize) -> usize {
    if phase_step <= 0.0 {
        return cap;
    }
    ((0.5 / phase_step).floor() as usize).clamp(1, cap)
}

fn band_limited_square(phase: f64, phase_step: f64) -> f64 {
    let limit = harmonic_limit(phase_step, 15);
    let mut sum = 0.0;
    for harmonic in (1..=limit).step_by(2) {
        sum += (TWO_PI * phase * harmonic as f64).sin() / harmonic as f64;
    }
    (sum * 4.0 / core::f64::consts::PI).clamp(-1.0, 1.0)
}

fn band_limited_triangle(phase: f64, phase_step: f64) -> f64 {
    let limit = harmonic_limit(phase_step, 15);
    let mut sum = 0.0;
    let mut sign = 1.0;
    for harmonic in (1..=limit).step_by(2) {
        let n = harmonic as f64;
        sum += sign * (TWO_PI * phase * n).sin() / (n * n);
        sign = -sign;
    }
    (sum * 8.0 / (core::f64::consts::PI * core::f64::consts::PI)).clamp(-1.0, 1.0)
}

fn band_limited_saw(phase: f64, phase_step: f64) -> f64 {
    let limit = harmonic_limit(phase_step, 12);
    let mut sum = 0.0;
    for harmonic in 1..=limit {
        let n = harmonic as f64;
        sum += (TWO_PI * phase * n).sin() / n;
    }
    (-sum * 2.0 / core::f64::consts::PI).clamp(-1.0, 1.0)
}

#[cfg(test)]
fn apply_two_pole_lowpass(data: &mut [i16], sample_rate: u32, cutoff_hz: f64) {
    if data.len() < 2 || sample_rate == 0 {
        return;
    }
    let mut lowpass = TwoPoleLowpass::new(sample_rate, cutoff_hz);
    for frame in data.chunks_exact_mut(2) {
        let filtered = lowpass.process([frame[0], frame[1]]);
        frame.copy_from_slice(&filtered);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smaf_player::{MaFmOperatorPatch, MaFmPatch};

    fn render_fm_note_reference(note: &MaFmNote, sample_rate: u32) -> Vec<i16> {
        let bend = (f64::from(note.pitch_bend.min(0x3fff)) - 8192.0) / 8192.0 * 2.0;
        let midi_note = f64::from(note.note) + f64::from(note.patch.note_shift) + bend;
        let frequency = 440.0 * 2.0_f64.powf((midi_note - 69.0) / 12.0);
        let mut voice = Voice::new(note.patch, frequency, midi_note, f64::from(sample_rate));
        let gate_frames = (u64::from(note.gate_ms) * u64::from(sample_rate) / 1000) as usize;
        let release_limit = (u64::from(MAX_RELEASE_MS) * u64::from(sample_rate) / 1000) as usize;
        let gain = f64::from(note.velocity) / 127.0 * (f64::from(note.volume) / 127.0) * (f64::from(note.expression) / 127.0) * FM_NOTE_HEADROOM;
        let channel_pan = note.pan.map(|value| (f64::from(value) - 64.0) / 63.0).unwrap_or(0.0);
        let pan = (channel_pan + f64::from(note.patch.default_pan)).clamp(-1.0, 1.0);
        let left_gain = if pan > 0.0 { 1.0 - pan } else { 1.0 };
        let right_gain = if pan < 0.0 { 1.0 + pan } else { 1.0 };

        let mut data = Vec::new();
        for _ in 0..gate_frames {
            let sample = voice.tick() * gain;
            data.push(float_to_i16(sample * left_gain));
            data.push(float_to_i16(sample * right_gain));
        }
        voice.key_off();
        for _ in 0..release_limit {
            if voice.finished() {
                break;
            }
            let sample = voice.tick() * gain;
            data.push(float_to_i16(sample * left_gain));
            data.push(float_to_i16(sample * right_gain));
        }
        apply_two_pole_lowpass(&mut data, sample_rate, OUTPUT_LOWPASS_HZ);
        data
    }

    #[test]
    fn linear_multi_zero_is_half_frequency_and_high_values_do_not_use_opl_doubling() {
        let base = 440.0;
        let sample_rate = 44_100.0;
        let note = 69.0;
        let half = MaFmOperatorPatch {
            multi: 0,
            ..MaFmOperatorPatch::default()
        };
        let fifteen = MaFmOperatorPatch {
            multi: 15,
            ..MaFmOperatorPatch::default()
        };
        let half_op = Operator::new(half, base, note, sample_rate);
        let high_op = Operator::new(fifteen, base, note, sample_rate);
        assert!((half_op.phase_step - 220.0 / sample_rate).abs() < 0.0001);
        assert!((high_op.phase_step - 6600.0 / sample_rate).abs() < 0.001);
    }

    #[test]
    fn zero_attack_rate_never_rises() {
        let patch = MaFmOperatorPatch {
            ar: 0,
            tl: 0,
            multi: 1,
            ..MaFmOperatorPatch::default()
        };
        let mut operator = Operator::new(patch, 440.0, 69.0, 44_100.0);
        for _ in 0..128 {
            assert_eq!(operator.tick(0.0, 0.0), 0.0);
        }
    }

    #[test]
    fn higher_ma_wave_selectors_use_the_documented_wave_families() {
        let phase = 0.173;
        let step = 440.0 / 44_100.0;
        assert_ne!(waveform(16, phase, step), waveform(0, phase, step));
        assert_ne!(waveform(26, phase, step), waveform(2, phase, step));
        assert_eq!(waveform(15, phase, step), 0.0);
        assert_eq!(waveform(23, phase, step), 0.0);
        assert_eq!(waveform(31, phase, step), 0.0);
    }

    #[test]
    fn rr_zero_modulator_does_not_force_a_false_five_second_tail() {
        let mut operators = [MaFmOperatorPatch::default(); 4];
        operators[0].rr = 0;
        operators[0].tl = 0;
        operators[1].rr = 15;
        operators[1].tl = 0;
        let patch = MaFmPatch {
            algorithm: 0,
            operator_count: 2,
            note_shift: 0,
            default_pan: 0.0,
            lfo: 0,
            operators,
        };
        let mut voice = Voice::new(patch, 440.0, 69.0, 44_100.0);
        for _ in 0..512 {
            voice.tick();
        }
        voice.key_off();
        for _ in 0..44_100 {
            voice.tick();
            if voice.finished() {
                return;
            }
        }
        panic!("audible carrier did not retire within one second");
    }

    #[test]
    fn handset_lowpass_reduces_near_nyquist_energy() {
        let mut data: Vec<i16> = Vec::new();
        for index in 0..2048 {
            let sample = if index & 1 == 0 { 30_000 } else { -30_000 };
            data.push(sample);
            data.push(sample);
        }
        let before: i64 = data.iter().map(|sample| i64::from(sample.abs())).sum();
        apply_two_pole_lowpass(&mut data, 44_100, OUTPUT_LOWPASS_HZ);
        let after: i64 = data.iter().map(|sample| i64::from(sample.abs())).sum();
        assert!(after < before / 2);
    }

    #[test]
    fn embedded_renderer_resamples_mono_wave_and_keeps_sf2_events_out() {
        let events = vec![
            (
                0,
                SmafEvent::Wave {
                    channels: 1,
                    sampling_rate: 8_000,
                    data: vec![0, 16_000, -16_000, 0],
                    dynamics: WaveDynamics::UNITY,
                },
            ),
            (
                0,
                SmafEvent::MidiNoteOn {
                    channel: 0,
                    note: 60,
                    velocity: 127,
                },
            ),
        ];
        let rendered = render_embedded_audio(&events, 16_000);
        assert_eq!(rendered.channels, 2);
        assert_eq!(rendered.sampling_rate, 16_000);
        assert_eq!(rendered.data.len(), 8 * 2);
        assert_eq!(rendered.data[0], rendered.data[1]);
        assert!(rendered.data.iter().any(|sample| *sample != 0));
    }

    #[test]
    fn wave_renderer_applies_velocity_volume_expression_and_pan() {
        let source = [16_000i16, 16_000];
        let rendered = render_wave_event(
            1,
            8_000,
            &source,
            WaveDynamics {
                velocity: 64,
                volume: 100,
                expression: 127,
                pan: Some(127),
            },
            8_000,
        );
        assert_eq!(rendered.channels, 2);
        assert_eq!(rendered.data.len(), 4);
        // Full-right pan suppresses the left channel. Velocity and channel
        // volume reduce the right channel without modifying decoded source PCM.
        assert_eq!(rendered.data[0], 0);
        assert!(rendered.data[1] > 5_000 && rendered.data[1] < 7_000);
        assert_eq!(source, [16_000, 16_000]);
    }

    #[test]
    fn renders_stereo_and_respects_gate() {
        let mut operators = [MaFmOperatorPatch::default(); 4];
        operators[0].tl = 20;
        operators[1].tl = 0;
        let note = MaFmNote {
            channel: 0,
            note: 69,
            velocity: 127,
            gate_ms: 100,
            volume: 127,
            expression: 127,
            pan: Some(64),
            pitch_bend: 8192,
            patch: MaFmPatch {
                algorithm: 0,
                operator_count: 2,
                note_shift: 0,
                default_pan: 0.0,
                lfo: 0,
                operators,
            },
        };
        let rendered = render_ma_fm_note(&note, 44_100);
        assert_eq!(rendered.channels, 2);
        assert_eq!(rendered.sampling_rate, 44_100);
        assert!(rendered.data.len() >= 4_410 * 2);
        assert!(rendered.data.iter().any(|sample| *sample != 0));
    }

    #[test]
    fn stateful_fm_is_sample_identical_to_the_previous_whole_note_renderer() {
        let mut operators = [MaFmOperatorPatch::default(); 4];
        operators[0].tl = 20;
        operators[1].tl = 0;
        operators[1].rr = 12;
        let note = MaFmNote {
            channel: 0,
            note: 69,
            velocity: 103,
            gate_ms: 37,
            volume: 111,
            expression: 96,
            pan: Some(81),
            pitch_bend: 8_400,
            patch: MaFmPatch {
                algorithm: 0,
                operator_count: 2,
                note_shift: 0,
                default_pan: -0.1,
                lfo: 2,
                operators,
            },
        };

        let expected = render_fm_note_reference(&note, 24_000);
        let actual = render_ma_fm_note(&note, 24_000);

        assert_eq!(actual.data, expected);
    }

    #[test]
    fn incremental_embedded_mix_is_sample_identical_across_overlapping_events() {
        let mut operators = [MaFmOperatorPatch::default(); 4];
        operators[0].tl = 18;
        operators[1].tl = 0;
        operators[1].rr = 15;
        let note = MaFmNote {
            channel: 0,
            note: 64,
            velocity: 100,
            gate_ms: 25,
            volume: 120,
            expression: 110,
            pan: Some(48),
            pitch_bend: 8192,
            patch: MaFmPatch {
                algorithm: 0,
                operator_count: 2,
                note_shift: 0,
                default_pan: 0.0,
                lfo: 0,
                operators,
            },
        };
        let events = vec![
            (3, SmafEvent::MaFmNote(note)),
            (
                10,
                SmafEvent::Wave {
                    channels: 1,
                    sampling_rate: 8_000,
                    data: vec![0, 4_000, 8_000, -4_000, 0],
                    dynamics: WaveDynamics {
                        velocity: 91,
                        volume: 100,
                        expression: 113,
                        pan: Some(90),
                    },
                },
            ),
        ];

        let sample_rate = 24_000;
        let mut reference = Vec::<f64>::new();
        for (time_ms, event) in &events {
            let start_frame = time_ms * sample_rate as usize / 1000;
            match event {
                SmafEvent::Wave {
                    channels,
                    sampling_rate,
                    data,
                    dynamics,
                } => mix_wave(&mut reference, start_frame, sample_rate, *channels, *sampling_rate, data, *dynamics),
                SmafEvent::MaFmNote(note) => {
                    let rendered = render_ma_fm_note(note, sample_rate);
                    let needed = start_frame.saturating_mul(2).saturating_add(rendered.data.len());
                    reference.resize(reference.len().max(needed), 0.0);
                    let start = start_frame * 2;
                    for (target, sample) in reference[start..start + rendered.data.len()].iter_mut().zip(rendered.data) {
                        *target += f64::from(sample) / 32768.0;
                    }
                }
                _ => unreachable!(),
            }
        }
        let expected: Vec<i16> = reference.into_iter().map(float_to_i16).collect();

        let actual = render_embedded_audio(&events, sample_rate);

        assert_eq!(actual.data, expected);
    }
}
