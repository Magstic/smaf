#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

use alloc::vec::Vec;
mod adpcm;
mod ma_voice;
mod pcm;

pub use ma_voice::{MaCustomVoice, MaFmNote, MaFmOperatorPatch, MaFmPatch, MaPcmRomVoice, MaVoiceKey};

use smaf::{
    Channel, ChannelStatus, ChannelType, PCMAudioSequenceEvent, PCMAudioTrack, PCMAudioTrackChunk, PCMDataChunk, ScoreTrack, ScoreTrackChunk,
    ScoreTrackSequenceEvent, Smaf, SmafChunk,
};

use self::{adpcm::decode_yamaha_adpcm, ma_voice::{parse_ma_voice_exclusive, parse_setup_custom_voices}, pcm::{decode_pcm, PcmEncoding}};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaveDynamics {
    pub velocity: u8,
    pub volume: u8,
    pub expression: u8,
    pub pan: Option<u8>,
}

impl WaveDynamics {
    pub const UNITY: Self = Self {
        velocity: 127,
        volume: 127,
        expression: 127,
        pan: None,
    };
}

#[derive(Clone, Debug, PartialEq)]
pub enum SmafEvent {
    Wave {
        channels: u8,
        sampling_rate: u32,
        data: Vec<i16>,
        dynamics: WaveDynamics,
    },
    MaFmNote(MaFmNote),
    MidiNoteOn { channel: u8, note: u8, velocity: u8 },
    MidiNoteOff { channel: u8, note: u8, velocity: u8 },
    MidiProgramChange { channel: u8, program: u8 },
    MidiControlChange { channel: u8, control: u8, value: u8 },
    MidiPitchBend { channel: u8, value: u16 },
    MidiSysEx(Vec<u8>),
    End,
}

pub fn parse_smaf(raw: &[u8]) -> Vec<(usize, SmafEvent)> {
    let Ok(smaf) = Smaf::parse(raw) else {
        return Vec::new();
    };

    let mut result = Vec::new();
    let mut handy_channel_offset = 0;
    let mut handy_tone_map = ToneMap::new();

    for chunk in &smaf.chunks {
        match chunk {
            SmafChunk::ScoreTrack(_, x) => {
                let (events, next_offset) = parse_score_track_events(x, handy_channel_offset, &mut handy_tone_map);
                result.extend(events);
                handy_channel_offset = next_offset;
            }
            SmafChunk::PCMAudioTrack(_, x) => result.extend(parse_pcm_audio_track_events(x)),
            SmafChunk::SoftbankSequenceData(x) => {
                let mut tone_map = ToneMap::new();
                tone_map.init_track(smaf::FormatType::HandyPhoneStandard, &[], handy_channel_offset);
                let (events, next_offset) = parse_sequence_events(x, 20, 20, handy_channel_offset, true, &[], &mut tone_map);
                result.extend(events);
                handy_channel_offset = next_offset;
            }
            _ => {}
        }
    }

    // Each track parser reports an End marker. Collapse them into a single
    // file-level End at the latest scheduled point so a loop cannot restart
    // while a gated wave or note-off from another track is still active.
    let end_time = result.iter().map(|(time, _)| *time).max().unwrap_or(0);
    result.retain(|(_, event)| !matches!(event, SmafEvent::End));
    result.push((end_time, SmafEvent::End));

    result.sort_by(|(left_time, left_event), (right_time, right_event)| {
        left_time
            .cmp(right_time)
            .then_with(|| event_sort_key(left_event).cmp(&event_sort_key(right_event)))
    });

    result
}

fn event_sort_key(event: &SmafEvent) -> (u8, [u8; 3]) {
    match event {
        SmafEvent::MidiSysEx(data) => (4, [data.first().copied().unwrap_or(0xf0), 0, 0]),
        SmafEvent::MidiControlChange { channel, control, value } => (5, [0xb0 | *channel, *control, *value]),
        SmafEvent::MidiPitchBend { channel, value } => (5, [0xe0 | *channel, (value & 0x7f) as u8, ((value >> 7) & 0x7f) as u8]),
        SmafEvent::MidiProgramChange { channel, program } => (6, [0xc0 | *channel, *program, 0]),
        SmafEvent::MidiNoteOff { channel, note, velocity } => (20, [0x80 | *channel, *note, *velocity]),
        SmafEvent::MidiNoteOn { channel, note, velocity } => (30, [0x90 | *channel, *note, *velocity]),
        SmafEvent::Wave { channels, .. } => (40, [*channels, 0, 0]),
        SmafEvent::MaFmNote(note) => (35, [note.channel, note.note, note.velocity]),
        SmafEvent::End => (99, [0xff, 0x2f, 0]),
    }
}

fn parse_score_track_events(track: &ScoreTrack, handy_channel_offset: u8, handy_tone_map: &mut ToneMap) -> (Vec<(usize, SmafEvent)>, u8) {
    let mut result = Vec::new();
    let mut mobile_tone_map = ToneMap::new();
    let pcm_chunks = track
        .chunks
        .iter()
        .find_map(|chunk| {
            if let ScoreTrackChunk::PCMData(x) = chunk {
                Some(x.as_slice())
            } else {
                None
            }
        })
        .unwrap_or(&[]);
    let is_handy = track.format_type == smaf::FormatType::HandyPhoneStandard;

    if is_handy {
        handy_tone_map.init_track(track.format_type, &track.channel_status, handy_channel_offset);
    } else {
        mobile_tone_map.init_track(track.format_type, &track.channel_status, 0);
    }

    let tone_map: &mut ToneMap = if is_handy { handy_tone_map } else { &mut mobile_tone_map };

    for setup_data in track
        .chunks
        .iter()
        .filter_map(|chunk| if let ScoreTrackChunk::SetupData(x) = chunk { Some(*x) } else { None })
    {
        tone_map.load_custom_voices(setup_data);
        result.extend(parse_setup_sysex_events(setup_data));
    }

    for sequence_data in track
        .chunks
        .iter()
        .filter_map(|chunk| if let ScoreTrackChunk::SequenceData(x) = chunk { Some(x) } else { None })
    {
        let (events, _) = parse_sequence_events(
            sequence_data,
            track.timebase_d,
            track.timebase_g,
            handy_channel_offset,
            is_handy,
            pcm_chunks,
            &mut *tone_map,
        );
        result.extend(events);
    }

    let next_offset = if is_handy {
        handy_channel_offset.saturating_add(4)
    } else {
        handy_channel_offset
    };

    (result, next_offset)
}

fn parse_sequence_events(
    sequence_data: &[smaf::SequenceData],
    timebase_d: u8,
    timebase_g: u8,
    channel_offset: u8,
    use_channel_offset: bool,
    pcm_chunks: &[PCMDataChunk<'_>],
    tone_map: &mut ToneMap,
) -> (Vec<(usize, SmafEvent)>, u8) {
    let mut result = Vec::new();
    let mut now = 0usize;
    let mut playback_end = 0usize;
    let mut octave_shift = [0i8; MAX_SMAF_CHANNELS];

    let map_channel = |channel: u8| {
        if use_channel_offset {
            channel.saturating_add(channel_offset)
        } else {
            channel
        }
    };

    for event in sequence_data.iter() {
        now = now.saturating_add((event.duration as usize).saturating_mul(timebase_d as usize));
        let time = now;
        playback_end = playback_end.max(time);

        match event.event {
            ScoreTrackSequenceEvent::NoteMessage {
                channel,
                note,
                velocity,
                gate_time,
            } => {
                let channel = map_channel(channel);
                // Mobile Standard Stream PCM is selected by Bank MSB 125.
                // Its Note Number maps to a WaveID; the MIDI source channel is
                // not part of the Mwa resource identity. Notes outside the
                // Stream-PCM ranges remain ordinary rhythm/melodic notes.
                let stream_wave_id = tone_map.stream_wave_id(channel, note);
                let pcm = stream_wave_id.and_then(|wave_number| {
                    pcm_chunks.iter().find_map(|pcm_chunk| {
                        let PCMDataChunk::WaveData(number, wave) = pcm_chunk;
                        if *number == wave_number {
                            Some(wave)
                        } else {
                            None
                        }
                    })
                });

                if let Some(pcm) = pcm {
                    let channels = channel_count(pcm.channel);
                    let decoded = match pcm.format {
                        smaf::StreamWaveFormat::YamahaADPCM => {
                            if pcm.base_bit != smaf::BaseBit::Bit4 {
                                continue;
                            }
                            decode_yamaha_adpcm(pcm.wave_data, channels)
                        }
                        smaf::StreamWaveFormat::TwosComplementPCM => {
                            decode_pcm(pcm.wave_data, pcm.base_bit, PcmEncoding::TwosComplement, channels)
                        }
                        smaf::StreamWaveFormat::OffsetBinaryPCM => {
                            decode_pcm(pcm.wave_data, pcm.base_bit, PcmEncoding::OffsetBinary, channels)
                        }
                    };
                    let Some(mut decoded) = decoded else {
                        continue;
                    };
                    trim_to_gate(&mut decoded, channels, pcm.sampling_freq as u32, gate_time, timebase_g);
                    let wave_ms = decoded_duration_ms(decoded.len(), channels, pcm.sampling_freq as u32);
                    playback_end = playback_end.max(time.saturating_add(wave_ms));
                    // Stream PCM is a note-driven MA voice. Its authored note
                    // velocity and channel dynamics apply even though the sample
                    // data itself bypasses the external MIDI/SF2 synth.
                    let dynamics = tone_map.wave_dynamics(channel, velocity);
                    result.push((
                        time,
                        SmafEvent::Wave {
                            channels,
                            sampling_rate: pcm.sampling_freq as _,
                            data: decoded,
                            dynamics,
                        },
                    ));
                } else if let Some(MaCustomVoice::Fm { patch, .. }) = tone_map.custom_voice(channel, note) {
                    // File-defined Yamaha MA custom FM voices are authoritative.
                    // Do not route them through GM/SF2: doing so discards the
                    // authored operator ratios, envelopes, algorithms and pan.
                    let gate_ms = gate_time.saturating_mul(timebase_g as u32);
                    let duration = gate_ms as usize;
                    let channel_index = (channel as usize).min(octave_shift.len() - 1);
                    let source_note =
                        (note as i16 + octave_shift[channel_index] as i16 * 12).clamp(0, 127) as u8;
                    let fm_note = tone_map.fm_note(channel, source_note, velocity, gate_ms, patch);
                    playback_end = playback_end.max(time.saturating_add(duration));
                    result.push((time, SmafEvent::MaFmNote(fm_note)));
                } else {
                    // A Bank-125 note whose referenced Mwa is absent is not
                    // silently discarded: it may be a file-defined PCM-ROM
                    // voice or a device-ROM preset. Those are intentionally
                    // left on the MIDI/SF2 approximation boundary because the
                    // Yamaha handset ROM is not present in the MMF.
                    let duration = gate_time.saturating_mul(timebase_g as u32) as usize;
                    let channel_index = (channel as usize).min(octave_shift.len() - 1);
                    let shifted_note = tone_map.map_note(channel, note as i16 + (octave_shift[channel_index] as i16 * 12));
                    let velocity = tone_map.note_velocity(channel, velocity);
                    let midi_channel = tone_map.real_channel(channel);
                    let duration = tone_map.note_duration(channel, duration);
                    playback_end = playback_end.max(time.saturating_add(duration));
                    result.push((
                        time,
                        SmafEvent::MidiNoteOn {
                            channel: midi_channel,
                            note: shifted_note,
                            velocity,
                        },
                    ));
                    result.push((
                        time + duration,
                        SmafEvent::MidiNoteOff {
                            channel: midi_channel,
                            note: shifted_note,
                            velocity: 0,
                        },
                    ));
                }
            }
            ScoreTrackSequenceEvent::ControlChange { channel, control, value } => {
                let channel = map_channel(channel);
                tone_map.update_control(channel, control, value);
                let value = tone_map.midi_control_value(control, value);
                let channel = tone_map.real_channel(channel);
                result.push((time, SmafEvent::MidiControlChange { channel, control, value }))
            }
            ScoreTrackSequenceEvent::ProgramChange { channel, program } => {
                let source_channel = map_channel(channel);
                let (channel, mapped_program) = tone_map.set_program(source_channel, program);
                result.push((
                    time,
                    SmafEvent::MidiProgramChange {
                        channel,
                        program: mapped_program,
                    },
                ));
            }
            ScoreTrackSequenceEvent::Exclusive(ref data) => {
                if let Some(voice) = parse_ma_voice_exclusive(data) {
                    // Inline VM35 definitions update the same authored voice
                    // table as Mtsu. They are not generic MIDI SysEx.
                    tone_map.custom_voices.push(voice);
                } else {
                    result.push((time, SmafEvent::MidiSysEx(make_sysex_message(data))));
                }
            }
            ScoreTrackSequenceEvent::Nop => continue,
            ScoreTrackSequenceEvent::PitchBend { channel, value } => {
                let channel = map_channel(channel);
                tone_map.set_pitch_bend(channel, value);
                let midi_channel = tone_map.real_channel(channel);
                result.push((
                    time,
                    SmafEvent::MidiPitchBend {
                        channel: midi_channel,
                        value: value.min(0x3fff),
                    },
                ));
            }
            ScoreTrackSequenceEvent::Volume { channel, value } => {
                let channel = map_channel(channel);
                tone_map.update_control(channel, 7, value);
                if tone_map.format_type == smaf::FormatType::HandyPhoneStandard {
                    if tone_map.is_rhythm(channel) {
                        result.push((
                            time,
                            SmafEvent::MidiControlChange {
                                channel: MIDI_DRUM_CHANNEL,
                                control: 7,
                                value: 100,
                            },
                        ));
                    } else {
                        let midi_channel = tone_map.real_channel(channel);
                        let value = tone_map.effective_volume(channel);
                        result.push((
                            time,
                            SmafEvent::MidiControlChange {
                                channel: midi_channel,
                                control: 7,
                                value,
                            },
                        ));
                    }
                } else {
                    let channel = tone_map.real_channel(channel);
                    result.push((time, SmafEvent::MidiControlChange { channel, control: 7, value }));
                }
            }
            ScoreTrackSequenceEvent::Pan { channel, value } => {
                let channel = map_channel(channel);
                tone_map.update_control(channel, 10, value);
                let midi_channel = tone_map.real_channel(channel);
                result.push((time, SmafEvent::MidiControlChange { channel: midi_channel, control: 10, value }));
            }
            ScoreTrackSequenceEvent::Expression { channel, value } => {
                let channel = map_channel(channel);
                if tone_map.format_type == smaf::FormatType::HandyPhoneStandard {
                    let value = tone_map.set_expression(channel, value);
                    if !tone_map.is_rhythm(channel) {
                        let channel = tone_map.real_channel(channel);
                        result.push((time, SmafEvent::MidiControlChange { channel, control: 7, value }));
                    }
                } else {
                    tone_map.update_control(channel, 11, value);
                    let midi_channel = tone_map.real_channel(channel);
                    result.push((time, SmafEvent::MidiControlChange { channel: midi_channel, control: 11, value }));
                }
            }
            ScoreTrackSequenceEvent::OctaveShift { channel, value } => {
                let channel = map_channel(channel);
                if let Some(value) = parse_octave_shift(value) {
                    let channel_index = (channel as usize).min(octave_shift.len() - 1);
                    octave_shift[channel_index] = value;
                }
            }
            ScoreTrackSequenceEvent::Modulation { channel, value } => {
                let channel = map_channel(channel);
                let channel = tone_map.real_channel(channel);
                result.push((time, SmafEvent::MidiControlChange { channel, control: 1, value }));
            }
            ScoreTrackSequenceEvent::BankSelect { channel, value } => {
                let channel = map_channel(channel);
                tone_map.update_bank_select(channel, value);
                let midi_channel = tone_map.real_channel(channel);
                result.push((
                    time,
                    SmafEvent::MidiControlChange {
                        channel: midi_channel,
                        control: 0,
                        value: tone_map.midi_control_value(0, value),
                    },
                ));
            }
        }
    }
    result.push((playback_end.max(now), SmafEvent::End));

    let next_offset = if use_channel_offset {
        channel_offset.saturating_add(4)
    } else {
        channel_offset
    };

    (result, next_offset)
}

const MIDI_DRUM_CHANNEL: u8 = 9;
const MAX_SMAF_CHANNELS: usize = 64;
const MELODY_ALLOCATION_ORDER: [u8; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15];

struct ToneMap {
    format_type: smaf::FormatType,
    channel_types: [u8; MAX_SMAF_CHANNELS],
    programs: [u8; MAX_SMAF_CHANNELS],
    source_programs: [u8; MAX_SMAF_CHANNELS],
    channel_volumes: [u8; MAX_SMAF_CHANNELS],
    expressions: [u8; MAX_SMAF_CHANNELS],
    velocities: [u8; MAX_SMAF_CHANNELS],
    bank_msb: [u8; MAX_SMAF_CHANNELS],
    bank_lsb: [u8; MAX_SMAF_CHANNELS],
    pan: [u8; MAX_SMAF_CHANNELS],
    pan_explicit: [bool; MAX_SMAF_CHANNELS],
    pitch_bend: [u16; MAX_SMAF_CHANNELS],
    forced_rhythm: [bool; MAX_SMAF_CHANNELS],
    real_map: [Option<u8>; MAX_SMAF_CHANNELS],
    custom_voices: Vec<MaCustomVoice>,
}

impl ToneMap {
    fn new() -> Self {
        Self {
            format_type: smaf::FormatType::MobileStandardNoCompress,
            channel_types: [2; MAX_SMAF_CHANNELS],
            programs: [0; MAX_SMAF_CHANNELS],
            source_programs: [0; MAX_SMAF_CHANNELS],
            channel_volumes: [100; MAX_SMAF_CHANNELS],
            expressions: [127; MAX_SMAF_CHANNELS],
            velocities: [64; MAX_SMAF_CHANNELS],
            bank_msb: [0; MAX_SMAF_CHANNELS],
            bank_lsb: [0; MAX_SMAF_CHANNELS],
            pan: [64; MAX_SMAF_CHANNELS],
            pan_explicit: [false; MAX_SMAF_CHANNELS],
            pitch_bend: [8192; MAX_SMAF_CHANNELS],
            forced_rhythm: [false; MAX_SMAF_CHANNELS],
            real_map: [None; MAX_SMAF_CHANNELS],
            custom_voices: Vec::new(),
        }
    }

    fn init_track(&mut self, format_type: smaf::FormatType, channel_statuses: &[ChannelStatus], channel_offset: u8) {
        self.format_type = format_type;

        if format_type == smaf::FormatType::HandyPhoneStandard {
            let base = (channel_offset as usize).min(MAX_SMAF_CHANNELS - 4);
            for local in 0..4 {
                let channel = base + local;
                self.channel_types[channel] = channel_statuses
                    .get(local)
                    .map(|status| match &status.channel_type {
                        ChannelType::Rhythm => 3,
                        _ => 1,
                    })
                    .unwrap_or(1);
                self.programs[channel] = 0;
                self.source_programs[channel] = 0;
                self.channel_volumes[channel] = 100;
                self.expressions[channel] = 127;
                self.velocities[channel] = 127;
                self.bank_msb[channel] = 0;
                self.bank_lsb[channel] = 0;
                self.pan[channel] = 64;
                self.pan_explicit[channel] = false;
                self.pitch_bend[channel] = 8192;
                self.forced_rhythm[channel] = false;
            }
            return;
        }

        self.channel_types = [2; MAX_SMAF_CHANNELS];
        self.programs = [0; MAX_SMAF_CHANNELS];
        self.source_programs = [0; MAX_SMAF_CHANNELS];
        self.channel_volumes = [100; MAX_SMAF_CHANNELS];
        self.expressions = [127; MAX_SMAF_CHANNELS];
        self.velocities = [64; MAX_SMAF_CHANNELS];
        self.bank_msb = [0; MAX_SMAF_CHANNELS];
        self.bank_lsb = [0; MAX_SMAF_CHANNELS];
        self.pan = [64; MAX_SMAF_CHANNELS];
        self.pan_explicit = [false; MAX_SMAF_CHANNELS];
        self.pitch_bend = [8192; MAX_SMAF_CHANNELS];
        self.forced_rhythm = [false; MAX_SMAF_CHANNELS];
        self.real_map = [None; MAX_SMAF_CHANNELS];
        self.custom_voices.clear();

        for (channel, status) in channel_statuses.iter().take(16).enumerate() {
            self.channel_types[channel] = match &status.channel_type {
                ChannelType::NoCare | ChannelType::NoMelody => 2,
                ChannelType::Melody => 1,
                ChannelType::Rhythm => 3,
            };
        }
    }

    fn pseudo_channel(&self, channel: u8) -> usize {
        if self.format_type == smaf::FormatType::HandyPhoneStandard {
            (channel as usize).min(MAX_SMAF_CHANNELS - 1)
        } else {
            (channel & 0x0f) as usize
        }
    }

    fn is_rhythm(&self, channel: u8) -> bool {
        let channel = self.pseudo_channel(channel);
        self.channel_types[channel] == 3 || self.forced_rhythm[channel] || self.bank_msb[channel] == 0x7d
    }

    fn real_channel(&mut self, channel: u8) -> u8 {
        let channel = self.pseudo_channel(channel);
        if self.is_rhythm(channel as u8) {
            return MIDI_DRUM_CHANNEL;
        }

        if let Some(real_channel) = self.real_map[channel] {
            return real_channel;
        }

        let mut used = [false; 16];
        used[MIDI_DRUM_CHANNEL as usize] = true;
        for real_channel in self.real_map.iter().flatten() {
            used[*real_channel as usize] = true;
        }
        let real_channel = MELODY_ALLOCATION_ORDER
            .iter()
            .copied()
            .find(|candidate| !used[*candidate as usize])
            .unwrap_or(channel as u8);
        self.real_map[channel] = Some(real_channel);
        real_channel
    }

    fn update_control(&mut self, channel: u8, control: u8, value: u8) {
        let channel = self.pseudo_channel(channel);
        match control {
            0 => self.bank_msb[channel] = value & 0x7f,
            32 => self.bank_lsb[channel] = value & 0x7f,
            7 => self.channel_volumes[channel] = value.min(0x7f),
            10 => {
                self.pan[channel] = value.min(0x7f);
                self.pan_explicit[channel] = true;
            }
            11 => self.expressions[channel] = value.min(0x7f),
            _ => {}
        }
    }

    fn update_bank_select(&mut self, channel: u8, value: u8) {
        let channel = self.pseudo_channel(channel);
        if value & 0x80 != 0 {
            self.forced_rhythm[channel] = true;
        }
        self.bank_msb[channel] = value & 0x7f;
    }

    fn set_program(&mut self, channel: u8, program: u8) -> (u8, u8) {
        let channel = self.pseudo_channel(channel);
        let source_program = program.min(0x7f);
        self.source_programs[channel] = source_program;
        if self.format_type == smaf::FormatType::HandyPhoneStandard && self.channel_types[channel] == 3 {
            self.programs[channel] = source_program;
            return (MIDI_DRUM_CHANNEL, 0);
        }

        let midi_program = if self.is_rhythm(channel as u8) {
            0
        } else {
            self.map_program(channel as u8, source_program)
        };
        self.programs[channel] = midi_program;

        (self.real_channel(channel as u8), midi_program)
    }

    fn map_program(&self, _channel: u8, program: u8) -> u8 {
        // Generic MIDI/SF2 is only a fallback for Yamaha preset/ROM voices.
        // Preserve the authored program number instead of applying scene-
        // specific substitutions. File-defined MA custom voices are handled
        // by the dedicated Yamaha voice-synthesis path, not by this mapper.
        program & 0x7f
    }

    fn stream_wave_id(&self, channel: u8, note: u8) -> Option<u8> {
        if self.format_type == smaf::FormatType::HandyPhoneStandard {
            return None;
        }
        let channel = self.pseudo_channel(channel);
        if self.bank_msb[channel] != 0x7d {
            return None;
        }

        match note {
            0..=12 => Some(note + 1),
            92..=110 => Some(note - 78),
            _ => None,
        }
    }

    fn load_custom_voices(&mut self, data: &[u8]) {
        self.custom_voices.extend(parse_setup_custom_voices(data));
    }

    fn custom_voice(&self, channel: u8, note: u8) -> Option<MaCustomVoice> {
        let channel = self.pseudo_channel(channel);
        let bank_msb = self.bank_msb[channel];
        let bank_lsb = self.bank_lsb[channel];
        let program = self.source_programs[channel];
        let mut melodic = None;
        for voice in self.custom_voices.iter().copied() {
            let key = voice.key();
            if key.bank_msb != bank_msb || key.bank_lsb != bank_lsb || key.program != program {
                continue;
            }
            if key.drum_note != 0 {
                if key.drum_note == note {
                    return Some(voice);
                }
            } else if melodic.is_none() {
                melodic = Some(voice);
            }
        }
        melodic
    }

    fn fm_note(&mut self, channel: u8, note: u8, velocity: Option<u8>, gate_ms: u32, patch: MaFmPatch) -> MaFmNote {
        let velocity = self.note_velocity(channel, velocity);
        let index = self.pseudo_channel(channel);
        MaFmNote {
            channel,
            note,
            velocity,
            gate_ms,
            volume: self.channel_volumes[index],
            expression: self.expressions[index],
            pan: self.pan_explicit[index].then_some(self.pan[index]),
            pitch_bend: self.pitch_bend[index],
            patch,
        }
    }

    fn wave_dynamics(&mut self, channel: u8, velocity: Option<u8>) -> WaveDynamics {
        let velocity = self.note_velocity(channel, velocity);
        let index = self.pseudo_channel(channel);
        WaveDynamics {
            velocity,
            volume: self.channel_volumes[index],
            expression: self.expressions[index],
            pan: self.pan_explicit[index].then_some(self.pan[index]),
        }
    }

    fn set_pitch_bend(&mut self, channel: u8, value: u16) {
        let index = self.pseudo_channel(channel);
        self.pitch_bend[index] = value.min(0x3fff);
    }

    fn midi_control_value(&self, control: u8, value: u8) -> u8 {
        if self.format_type != smaf::FormatType::HandyPhoneStandard && matches!(control, 0 | 32) {
            // MA-3/MA-5 use MSB 124 for normal voices and 125 for rhythm /
            // Stream PCM. Generic SoundFont synths generally expect GM bank 0;
            // keep Yamaha bank state internally and expose only a canonical GM
            // bank on the fallback MIDI boundary.
            0
        } else {
            value & 0x7f
        }
    }

    fn map_note(&self, channel: u8, note: i16) -> u8 {
        let channel_index = self.pseudo_channel(channel);

        if self.format_type == smaf::FormatType::HandyPhoneStandard {
            if self.is_rhythm(channel) {
                return self.programs[channel_index].min(0x7f);
            }
            return (note + 36).clamp(0, 127) as u8;
        }

        let note = note.clamp(0, 127) as u8;
        if !self.is_rhythm(channel) {
            return note;
        }

        match note {
            0x12 => 45,
            0x1a => 41,
            0x1f => 47,
            0x4d => 50,
            0x54 => 43,
            0x59 => 48,
            _ => note,
        }
    }

    fn note_velocity(&mut self, channel: u8, velocity: Option<u8>) -> u8 {
        let channel = self.pseudo_channel(channel);
        if self.format_type == smaf::FormatType::HandyPhoneStandard && self.channel_types[channel] == 3 {
            return self.hps_drum_velocity(channel);
        }
        if let Some(velocity) = velocity {
            let velocity = velocity.min(0x7f);
            self.velocities[channel] = velocity;
            velocity
        } else {
            self.velocities[channel]
        }
    }

    fn hps_drum_velocity(&self, channel: usize) -> u8 {
        let value = if self.expressions[channel] < 64 {
            (self.channel_volumes[channel] as u16 * self.expressions[channel] as u16) / 102
        } else {
            (self.channel_volumes[channel] as u16 * self.expressions[channel] as u16) / 100
        };

        value.clamp(1, 127) as u8
    }

    fn set_expression(&mut self, channel: u8, value: u8) -> u8 {
        let channel = self.pseudo_channel(channel);
        self.expressions[channel] = value.min(0x7f);
        ((self.channel_volumes[channel] as u16 * self.expressions[channel] as u16) / 127).min(127) as u8
    }

    fn effective_volume(&self, channel: u8) -> u8 {
        let channel = self.pseudo_channel(channel);
        ((self.channel_volumes[channel] as u16 * self.expressions[channel] as u16) / 127).min(127) as u8
    }

    fn note_duration(&self, _channel: u8, duration: usize) -> usize {
        duration
    }

}

fn parse_setup_sysex_events(data: &[u8]) -> Vec<(usize, SmafEvent)> {
    let mut result = Vec::new();
    let mut offset = 0;

    while offset < data.len() {
        if data[offset] != 0xf0 {
            break;
        }
        offset += 1;

        let Some((length, next_offset)) = read_midi_vlq(data, offset) else {
            break;
        };
        offset = next_offset;

        if offset + length > data.len() {
            break;
        }

        let payload = &data[offset..offset + length];
        // MA custom-voice definitions configure this engine's internal FM/PCM
        // voice table. Forwarding them to an unrelated GM/SF2 synth is both
        // ineffective and can perturb synth-specific SysEx state. Preserve
        // only non-voice setup messages (reset/effects/etc.) at the MIDI edge.
        if parse_ma_voice_exclusive(payload).is_none() {
            result.push((0, SmafEvent::MidiSysEx(make_sysex_message(payload))));
        }
        offset += length;
    }

    result
}

fn read_midi_vlq(data: &[u8], mut offset: usize) -> Option<(usize, usize)> {
    let mut value = 0usize;
    loop {
        let byte = *data.get(offset)?;
        offset += 1;
        value = (value << 7) | ((byte & 0x7f) as usize);
        if byte & 0x80 == 0 {
            return Some((value, offset));
        }
    }
}

fn make_sysex_message(data: &[u8]) -> Vec<u8> {
    let mut message = Vec::with_capacity(data.len() + 2);
    if data.first().copied() != Some(0xf0) {
        message.push(0xf0);
    }
    message.extend_from_slice(data);
    if message.last().copied() != Some(0xf7) {
        message.push(0xf7);
    }
    message
}

fn parse_octave_shift(value: u8) -> Option<i8> {
    match value {
        0x00..=0x04 => Some(value as i8),
        0x81..=0x84 => Some(-((value - 0x80) as i8)),
        _ => None,
    }
}

fn channel_count(channel: Channel) -> u8 {
    match channel {
        Channel::Mono => 1,
        Channel::Stereo => 2,
    }
}

fn trim_to_gate(data: &mut Vec<i16>, channels: u8, sampling_rate: u32, gate_time: u32, timebase_g: u8) {
    if channels == 0 || gate_time == 0 {
        data.clear();
        return;
    }

    let gate_ms = u64::from(gate_time) * u64::from(timebase_g);
    let max_frames = gate_ms.saturating_mul(u64::from(sampling_rate)) / 1000;
    let max_samples = max_frames
        .saturating_mul(u64::from(channels))
        .min(usize::MAX as u64) as usize;
    if data.len() > max_samples {
        data.truncate(max_samples);
    }
}

fn decoded_duration_ms(samples: usize, channels: u8, sampling_rate: u32) -> usize {
    if channels == 0 || sampling_rate == 0 {
        return 0;
    }
    let frames = samples / channels as usize;
    let rate = sampling_rate as usize;
    frames.saturating_mul(1000).saturating_add(rate - 1) / rate
}

fn decode_pcm_audio_track_wave(track: &PCMAudioTrack, data: &[u8]) -> Option<Vec<i16>> {
    let channels = channel_count(track.channel);
    match track.format {
        smaf::PcmWaveFormat::TwosComplementPCM => decode_pcm(data, track.base_bit, PcmEncoding::TwosComplement, channels),
        smaf::PcmWaveFormat::Adpcm => {
            if track.base_bit != smaf::BaseBit::Bit4 {
                return None;
            }
            decode_yamaha_adpcm(data, channels)
        }
        smaf::PcmWaveFormat::TwinVQ | smaf::PcmWaveFormat::MP3 => None,
    }
}

fn parse_pcm_audio_track_events(track: &PCMAudioTrack) -> Vec<(usize, SmafEvent)> {
    let Some(sequence_data) = track.chunks.iter().find_map(|chunk| {
        if let PCMAudioTrackChunk::SequenceData(x) = chunk {
            Some(x)
        } else {
            None
        }
    }) else {
        return Vec::new();
    };

    let mut result = Vec::new();
    let mut now = 0usize;
    let mut playback_end = 0usize;
    // PCM Audio Track control messages are local to its four audio channels.
    // Unlike Mobile Standard note velocity, WaveMessage has no velocity field,
    // so it uses unity velocity and the authored Volume/Expression/Pan state.
    let mut volume = [127u8; 4];
    let mut expression = [127u8; 4];
    let mut pan = [64u8; 4];
    let mut pan_explicit = [false; 4];

    for event in sequence_data.iter() {
        now = now.saturating_add((event.duration as usize).saturating_mul(track.timebase_d as usize));
        let time = now;
        playback_end = playback_end.max(time);

        match event.event {
            PCMAudioSequenceEvent::WaveMessage {
                channel,
                wave_number,
                gate_time,
            } => {
                let Some(pcm) = track.chunks.iter().find_map(|chunk| {
                    if let PCMAudioTrackChunk::WaveData(number, data) = chunk {
                        if *number == wave_number {
                            return Some(*data);
                        }
                    }
                    None
                }) else {
                    continue;
                };

                let Some(mut decoded) = decode_pcm_audio_track_wave(track, pcm) else {
                    continue;
                };
                let channels = channel_count(track.channel);
                trim_to_gate(&mut decoded, channels, track.sampling_freq as u32, gate_time, track.timebase_g);
                let wave_ms = decoded_duration_ms(decoded.len(), channels, track.sampling_freq as u32);
                playback_end = playback_end.max(time.saturating_add(wave_ms));
                let index = usize::from(channel.min(3));
                result.push((
                    time,
                    SmafEvent::Wave {
                        channels,
                        sampling_rate: track.sampling_freq as _,
                        data: decoded,
                        dynamics: WaveDynamics {
                            velocity: 127,
                            volume: volume[index],
                            expression: expression[index],
                            pan: pan_explicit[index].then_some(pan[index]),
                        },
                    },
                ));
            }
            PCMAudioSequenceEvent::Volume { channel, value } => {
                volume[usize::from(channel.min(3))] = value.min(127);
            }
            PCMAudioSequenceEvent::Expression { channel, value } => {
                expression[usize::from(channel.min(3))] = value.min(127);
            }
            PCMAudioSequenceEvent::Pan { channel, value } => {
                let index = usize::from(channel.min(3));
                pan[index] = value.min(127);
                pan_explicit[index] = true;
            }
            // Pitch Bend for sampled audio implies resampling/pitch shift. ZIC2's
            // PCM-Audio SFX corpus does not author it; keep it explicit rather
            // than applying an unverified rate law.
            PCMAudioSequenceEvent::PitchBend { .. }
            | PCMAudioSequenceEvent::Nop
            | PCMAudioSequenceEvent::Exclusive(_) => continue,
        }
    }

    result.push((playback_end.max(now), SmafEvent::End));
    result
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::{parse_pcm_audio_track_events, parse_sequence_events, parse_smaf, SmafEvent, ToneMap};
    use smaf::{
        BaseBit, Channel, ChannelStatus, ChannelType, PCMAudioSequenceData, PCMAudioSequenceEvent, PCMAudioTrack, PCMAudioTrackChunk, PcmWaveFormat,
        ScoreTrackSequenceEvent, SequenceData,
    };

    fn channel_status(channel_type: ChannelType) -> ChannelStatus {
        ChannelStatus {
            kcs: 0,
            vs: 0,
            led: 0,
            channel_type,
        }
    }

    #[test]
    fn preserves_yamaha_program_number_for_generic_gm_fallback() {
        let mut tone_map = ToneMap::new();
        tone_map.update_control(1, 0, 0x7c);
        tone_map.update_control(1, 32, 0x01);
        assert_eq!(tone_map.set_program(1, 0x22), (0, 0x22));
        assert_eq!(tone_map.midi_control_value(0, 0x7c), 0);
        assert_eq!(tone_map.midi_control_value(32, 0x01), 0);
    }

    #[test]
    fn maps_yamaha_rhythm_bank_to_midi_drum_channel() {
        let mut tone_map = ToneMap::new();
        tone_map.update_control(9, 0, 0x7d);
        tone_map.update_control(9, 32, 0x00);
        assert_eq!(tone_map.set_program(9, 0x02), (9, 0));
        assert_eq!(tone_map.map_note(9, 0x1a), 41);
    }

    #[test]
    fn maps_mobile_stream_pcm_notes_to_wave_ids_only_in_bank_125() {
        let mut tone_map = ToneMap::new();
        tone_map.init_track(smaf::FormatType::MobileStandardNoCompress, &[], 0);
        tone_map.update_control(9, 0, 0x7d);

        assert_eq!(tone_map.stream_wave_id(9, 0), Some(1));
        assert_eq!(tone_map.stream_wave_id(9, 12), Some(13));
        assert_eq!(tone_map.stream_wave_id(9, 92), Some(14));
        assert_eq!(tone_map.stream_wave_id(9, 110), Some(32));
        assert_eq!(tone_map.stream_wave_id(9, 13), None);
        assert_eq!(tone_map.stream_wave_id(9, 91), None);
        assert_eq!(tone_map.stream_wave_id(9, 111), None);

        tone_map.update_control(9, 0, 0x7c);
        assert_eq!(tone_map.stream_wave_id(9, 0), None);
    }

    #[test]
    fn compacts_melody_channels_around_midi_drum_channel() {
        let mut tone_map = ToneMap::new();
        assert_eq!(tone_map.real_channel(1), 0);
        assert_eq!(tone_map.real_channel(2), 1);
        assert_eq!(tone_map.real_channel(9), 2);
        tone_map.update_control(10, 0, 0x7d);
        assert_eq!(tone_map.real_channel(10), 9);
        assert_eq!(tone_map.real_channel(11), 3);
    }

    #[test]
    fn reuses_previous_velocity_for_mobile_notes_without_velocity() {
        let mut tone_map = ToneMap::new();
        tone_map.init_track(smaf::FormatType::MobileStandardNoCompress, &[], 0);
        let sequence = [
            SequenceData {
                duration: 0,
                event: ScoreTrackSequenceEvent::NoteMessage {
                    channel: 0,
                    note: 60,
                    velocity: Some(96),
                    gate_time: 10,
                },
            },
            SequenceData {
                duration: 0,
                event: ScoreTrackSequenceEvent::NoteMessage {
                    channel: 0,
                    note: 62,
                    velocity: None,
                    gate_time: 10,
                },
            },
        ];

        let (events, _) = parse_sequence_events(&sequence, 1, 1, 0, false, &[], &mut tone_map);
        assert!(events
            .iter()
            .any(|(_, event)| matches!(event, SmafEvent::MidiNoteOn { note: 62, velocity: 96, .. })));
    }

    #[test]
    fn applies_sequence_duration_before_event() {
        let mut tone_map = ToneMap::new();
        tone_map.init_track(smaf::FormatType::MobileStandardNoCompress, &[], 0);
        let sequence = [SequenceData {
            duration: 5,
            event: ScoreTrackSequenceEvent::NoteMessage {
                channel: 0,
                note: 60,
                velocity: Some(64),
                gate_time: 2,
            },
        }];

        let (events, _) = parse_sequence_events(&sequence, 4, 4, 0, false, &[], &mut tone_map);
        assert!(events
            .iter()
            .any(|(time, event)| *time == 20 && matches!(event, SmafEvent::MidiNoteOn { note: 60, .. })));
        assert!(events
            .iter()
            .any(|(time, event)| *time == 28 && matches!(event, SmafEvent::MidiNoteOff { note: 60, .. })));
    }

    #[test]
    fn applies_pcm_sequence_duration_before_event() {
        let track = PCMAudioTrack {
            format_type: 0,
            sequence_type: 0,
            channel: Channel::Mono,
            format: PcmWaveFormat::Adpcm,
            sampling_freq: 8000,
            base_bit: BaseBit::Bit4,
            timebase_d: 4,
            timebase_g: 4,
            chunks: vec![PCMAudioTrackChunk::SequenceData(vec![PCMAudioSequenceData {
                duration: 5,
                event: PCMAudioSequenceEvent::Nop,
            }])],
        };

        let events = parse_pcm_audio_track_events(&track);
        assert!(events.iter().any(|(time, event)| *time == 20 && matches!(event, SmafEvent::End)));
    }

    #[test]
    fn pcm_wave_gate_time_truncates_decoded_audio() {
        let wave_data = [0u8; 100];
        let track = PCMAudioTrack {
            format_type: 0,
            sequence_type: 0,
            channel: Channel::Mono,
            format: PcmWaveFormat::Adpcm,
            sampling_freq: 1000,
            base_bit: BaseBit::Bit4,
            timebase_d: 1,
            timebase_g: 1,
            chunks: vec![
                PCMAudioTrackChunk::SequenceData(vec![PCMAudioSequenceData {
                    duration: 0,
                    event: PCMAudioSequenceEvent::WaveMessage {
                        channel: 0,
                        wave_number: 1,
                        gate_time: 10,
                    },
                }]),
                PCMAudioTrackChunk::WaveData(1, &wave_data),
            ],
        };

        let events = parse_pcm_audio_track_events(&track);
        let wave = events.iter().find_map(|(_, event)| {
            if let SmafEvent::Wave { data, .. } = event {
                Some(data)
            } else {
                None
            }
        });
        assert_eq!(wave.unwrap().len(), 10);
        assert!(events.iter().any(|(time, event)| *time == 10 && matches!(event, SmafEvent::End)));
    }

    #[test]
    fn pcm_audio_track_preserves_authored_wave_dynamics() {
        let wave_data = [0u8; 16];
        let track = PCMAudioTrack {
            format_type: 0,
            sequence_type: 0,
            channel: Channel::Mono,
            format: PcmWaveFormat::Adpcm,
            sampling_freq: 8000,
            base_bit: BaseBit::Bit4,
            timebase_d: 1,
            timebase_g: 1,
            chunks: vec![
                PCMAudioTrackChunk::SequenceData(vec![
                    PCMAudioSequenceData { duration: 0, event: PCMAudioSequenceEvent::Volume { channel: 0, value: 96 } },
                    PCMAudioSequenceData { duration: 0, event: PCMAudioSequenceEvent::Expression { channel: 0, value: 80 } },
                    PCMAudioSequenceData { duration: 0, event: PCMAudioSequenceEvent::Pan { channel: 0, value: 127 } },
                    PCMAudioSequenceData {
                        duration: 0,
                        event: PCMAudioSequenceEvent::WaveMessage { channel: 0, wave_number: 1, gate_time: 2 },
                    },
                ]),
                PCMAudioTrackChunk::WaveData(1, &wave_data),
            ],
        };

        let events = parse_pcm_audio_track_events(&track);
        let dynamics = events.iter().find_map(|(_, event)| match event {
            SmafEvent::Wave { dynamics, .. } => Some(*dynamics),
            _ => None,
        }).unwrap();
        assert_eq!(dynamics.velocity, 127);
        assert_eq!(dynamics.volume, 96);
        assert_eq!(dynamics.expression, 80);
        assert_eq!(dynamics.pan, Some(127));
    }

    #[test]
    fn reference_mmf_matches_yamaha_adpcm_and_gate_timing() {
        let events = parse_smaf(include_bytes!("../../test_data/wave.mmf"));
        let (time, channels, sampling_rate, data) = events
            .iter()
            .find_map(|(time, event)| match event {
                SmafEvent::Wave {
                    channels,
                    sampling_rate,
                    data,
                    ..
                } => Some((*time, *channels, *sampling_rate, data)),
                _ => None,
            })
            .expect("reference MMF must contain a wave event");

        assert_eq!(time, 8);
        assert_eq!(channels, 1);
        assert_eq!(sampling_rate, 8000);
        // 673 gate ticks * 4 ms * 8000 Hz = 21,536 mono samples.
        assert_eq!(data.len(), 21_536);
        assert_eq!(
            &data[..32],
            &[
                -111, -253, -272, -119, -18, -182, -291, -76, -169, -253, -228, -295, -195, -177, -257, -115,
                -96, -113, -2, -81, 30, 236, 267, 352, 225, -26, 229, 131, 43, 69, -96, 9,
            ]
        );
    }

    #[test]
    fn hps_tracks_keep_independent_midi_channel_allocations() {
        let mut tone_map = ToneMap::new();
        let first_track = [channel_status(ChannelType::Melody)];
        tone_map.init_track(smaf::FormatType::HandyPhoneStandard, &first_track, 0);
        let first_sequence = [SequenceData {
            duration: 0,
            event: ScoreTrackSequenceEvent::ProgramChange { channel: 0, program: 40 },
        }];

        let (events, _) = parse_sequence_events(&first_sequence, 1, 1, 0, true, &[], &mut tone_map);
        assert!(events
            .iter()
            .any(|(_, event)| matches!(event, SmafEvent::MidiProgramChange { channel: 0, program: 40 })));

        let second_track = [channel_status(ChannelType::Melody)];
        tone_map.init_track(smaf::FormatType::HandyPhoneStandard, &second_track, 4);
        let second_sequence = [SequenceData {
            duration: 0,
            event: ScoreTrackSequenceEvent::ProgramChange { channel: 0, program: 41 },
        }];

        let (events, _) = parse_sequence_events(&second_sequence, 1, 1, 4, true, &[], &mut tone_map);
        assert!(events
            .iter()
            .any(|(_, event)| matches!(event, SmafEvent::MidiProgramChange { channel: 1, program: 41 })));
    }

    #[test]
    fn hps_rhythm_uses_program_as_drum_key_and_expression_velocity() {
        let mut tone_map = ToneMap::new();
        let statuses = [channel_status(ChannelType::Rhythm)];
        tone_map.init_track(smaf::FormatType::HandyPhoneStandard, &statuses, 0);
        let sequence = [
            SequenceData {
                duration: 0,
                event: ScoreTrackSequenceEvent::ProgramChange { channel: 0, program: 35 },
            },
            SequenceData {
                duration: 0,
                event: ScoreTrackSequenceEvent::Expression { channel: 0, value: 92 },
            },
            SequenceData {
                duration: 0,
                event: ScoreTrackSequenceEvent::NoteMessage {
                    channel: 0,
                    note: 1,
                    velocity: None,
                    gate_time: 10,
                },
            },
        ];

        let (events, _) = parse_sequence_events(&sequence, 1, 1, 0, true, &[], &mut tone_map);
        assert!(events
            .iter()
            .any(|(_, event)| matches!(event, SmafEvent::MidiProgramChange { channel: 9, program: 0 })));
        assert!(events.iter().any(|(_, event)| {
            matches!(
                event,
                SmafEvent::MidiNoteOn {
                    channel: 9,
                    note: 35,
                    velocity: 92
                }
            )
        }));
        assert!(!events
            .iter()
            .any(|(_, event)| matches!(event, SmafEvent::MidiControlChange { channel: 9, control: 11, .. })));
    }

    #[test]
    fn hps_melody_expression_is_folded_into_volume() {
        let mut tone_map = ToneMap::new();
        let statuses = [channel_status(ChannelType::Melody)];
        tone_map.init_track(smaf::FormatType::HandyPhoneStandard, &statuses, 0);
        let sequence = [
            SequenceData {
                duration: 0,
                event: ScoreTrackSequenceEvent::Volume { channel: 0, value: 100 },
            },
            SequenceData {
                duration: 0,
                event: ScoreTrackSequenceEvent::Expression { channel: 0, value: 92 },
            },
        ];

        let (events, _) = parse_sequence_events(&sequence, 1, 1, 0, true, &[], &mut tone_map);
        assert!(events.iter().any(|(_, event)| matches!(
            event,
            SmafEvent::MidiControlChange {
                channel: 0,
                control: 7,
                value: 72
            }
        )));
        assert!(!events
            .iter()
            .any(|(_, event)| matches!(event, SmafEvent::MidiControlChange { channel: 0, control: 11, .. })));
    }

    #[test]
    fn hps_melody_pitch_adds_base_offset_but_rhythm_does_not() {
        let mut melody_map = ToneMap::new();
        let melody = [channel_status(ChannelType::Melody)];
        melody_map.init_track(smaf::FormatType::HandyPhoneStandard, &melody, 0);
        assert_eq!(melody_map.map_note(0, 24), 60);

        let mut rhythm_map = ToneMap::new();
        let rhythm = [channel_status(ChannelType::Rhythm)];
        rhythm_map.init_track(smaf::FormatType::HandyPhoneStandard, &rhythm, 0);
        assert_eq!(rhythm_map.set_program(0, 38), (9, 0));
        assert_eq!(rhythm_map.map_note(0, 24), 38);
    }

    #[test]
    fn zic2_real_sfx_corpus_has_expected_wave_timing_and_rates() {
        use std::{fs, path::PathBuf};

        // (file number, sample rate, post-GateTime PCM sample count,
        //  wave start ms, file-level End ms)
        const EXPECTED: &[(u8, u32, usize, usize, usize)] = &[
            (0, 8000, 6336, 0, 792),
            (1, 8000, 8000, 0, 1000),
            (2, 8000, 10450, 0, 1308),
            (3, 12000, 5894, 0, 492),
            (4, 12000, 5464, 0, 456),
            (5, 8000, 1870, 0, 236),
            (6, 8000, 5014, 0, 628),
            (7, 12000, 8438, 0, 704),
            (8, 8000, 3936, 0, 492),
            (9, 12000, 4800, 0, 400),
            (10, 8000, 2846, 8, 368),
            (11, 8000, 3354, 8, 432),
            (12, 8000, 9226, 0, 1156),
            (13, 8000, 12704, 0, 1588),
            (14, 12000, 5982, 0, 500),
            (15, 8000, 2352, 0, 296),
            (16, 8000, 7156, 0, 896),
            (17, 8000, 1184, 8, 160),
            (18, 12000, 8438, 0, 704),
            (19, 8000, 8860, 0, 1108),
            (20, 8000, 3072, 8, 396),
            (21, 8000, 4982, 0, 624),
            (22, 8000, 1184, 8, 160),
            (23, 16000, 10560, 0, 660),
            (24, 12000, 10582, 0, 884),
            (25, 8000, 6528, 0, 816),
            (26, 12000, 17486, 0, 1460),
            (27, 8000, 5904, 0, 740),
            (28, 8000, 7530, 0, 944),
            (29, 12000, 7956, 0, 664),
            (30, 8000, 3566, 0, 448),
            (31, 8000, 6080, 0, 760),
            (32, 12000, 15620, 0, 1304),
        ];

        let corpus = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test_data/zic2_sfx");
        for &(number, expected_rate, expected_samples, expected_start, expected_end) in EXPECTED {
            let data = fs::read(corpus.join(std::format!("{number}.mmf"))).unwrap();
            let events = parse_smaf(&data);
            let waves: alloc::vec::Vec<_> = events
                .iter()
                .filter_map(|(time, event)| match event {
                    SmafEvent::Wave {
                        channels,
                        sampling_rate,
                        data,
                        ..
                    } => Some((*time, *channels, *sampling_rate, data)),
                    _ => None,
                })
                .collect();

            assert_eq!(waves.len(), 1, "{number}.mmf wave count");
            assert_eq!(waves[0].0, expected_start, "{number}.mmf wave start");
            assert_eq!(waves[0].1, 1, "{number}.mmf channel count");
            assert_eq!(waves[0].2, expected_rate, "{number}.mmf sample rate");
            assert_eq!(waves[0].3.len(), expected_samples, "{number}.mmf gated sample count");
            assert!(matches!(events.last(), Some((time, SmafEvent::End)) if *time == expected_end), "{number}.mmf End");
        }
    }

    #[test]
    fn zic2_atr_long_tail_is_cut_at_gate_time() {
        use std::{fs, path::PathBuf};

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test_data/zic2_sfx/17.mmf");
        let data = fs::read(path).unwrap();
        let events = parse_smaf(&data);
        let wave = events
            .iter()
            .find_map(|(_, event)| match event {
                SmafEvent::Wave { data, .. } => Some(data),
                _ => None,
            })
            .unwrap();

        // Awa1 physically decodes to 5288 samples. GateTime is 37 * 4 ms at
        // 8 kHz, therefore only 1184 samples are part of the audible event.
        assert_eq!(wave.len(), 1184);
        assert_eq!(
            &wave[..32],
            &[
                15, 0, 47, 32, 79, 64, 79, 0, 79, 317, 51, 426, 154, 202, 333, 215,
                745, 322, 854, 239, 648, 281, 479, 1012, 515, 578, 749, 493, 908, 522, 869, 913,
            ]
        );
    }


    #[test]
    fn zic2_bgm_stream_pcm_uses_bank_125_wave_id_mapping() {
        use std::{fs, path::PathBuf};

        const EXPECTED_COUNTS: &[(&str, usize)] = &[
            ("B0.mmf", 30),
            ("B1.mmf", 0),
            ("B2.mmf", 0),
            ("B3.mmf", 12),
            ("B4.mmf", 9),
            ("B5.mmf", 0),
            ("B6.mmf", 1),
            ("B7.mmf", 1),
            ("B8.mmf", 1),
            ("B9.mmf", 1),
        ];

        let corpus = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test_data/zic2_bgm");
        for &(name, expected_count) in EXPECTED_COUNTS {
            let data = fs::read(corpus.join(name)).unwrap();
            let events = parse_smaf(&data);
            let waves: alloc::vec::Vec<_> = events
                .iter()
                .filter_map(|(time, event)| match event {
                    SmafEvent::Wave {
                        channels,
                        sampling_rate,
                        data,
                        ..
                    } => Some((*time, *channels, *sampling_rate, data.len())),
                    _ => None,
                })
                .collect();
            assert_eq!(waves.len(), expected_count, "{name} sampled wave count");
        }

        // B0 uses MIDI channel 9 for both percussion and Stream PCM. Note 0
        // selects Mwa1 and must produce a 4 kHz sampled wave; ordinary drum
        // notes on the same channel/bank must remain MIDI percussion.
        let b0 = fs::read(corpus.join("B0.mmf")).unwrap();
        let events = parse_smaf(&b0);
        let first_wave = events
            .iter()
            .find_map(|(time, event)| match event {
                SmafEvent::Wave {
                    channels,
                    sampling_rate,
                    data,
                    dynamics,
                } => Some((*time, *channels, *sampling_rate, data.len(), *dynamics)),
                _ => None,
            })
            .unwrap();
        assert_eq!((first_wave.0, first_wave.1, first_wave.2, first_wave.3), (0, 1, 4000, 2256));
        assert_eq!(first_wave.4.velocity, 40);
        assert_eq!(first_wave.4.volume, 100);
        assert_eq!(first_wave.4.expression, 127);
        assert_eq!(first_wave.4.pan, None);
        assert!(events.iter().any(|(time, event)| {
            *time == 0
                && matches!(
                    event,
                    SmafEvent::MidiNoteOn {
                        channel: 9,
                        note: 15,
                        ..
                    }
                )
        }));
    }

    #[test]
    fn zic2_custom_fm_notes_bypass_generic_soundfont() {
        use std::{fs, path::PathBuf};

        const EXPECTED_FM_NOTES: &[(&str, usize)] = &[
            ("B0.mmf", 876),
            ("B1.mmf", 1778),
            ("B2.mmf", 653),
            ("B3.mmf", 421),
            ("B4.mmf", 358),
            ("B5.mmf", 408),
            ("B6.mmf", 0),
            ("B7.mmf", 0),
            ("B8.mmf", 0),
            ("B9.mmf", 0),
        ];

        let corpus = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test_data/zic2_bgm");
        for &(name, expected) in EXPECTED_FM_NOTES {
            let data = fs::read(corpus.join(name)).unwrap();
            let events = parse_smaf(&data);
            let count = events
                .iter()
                .filter(|(_, event)| matches!(event, SmafEvent::MaFmNote(_)))
                .count();
            assert_eq!(count, expected, "{name} custom FM note count");
        }

        let b0 = fs::read(corpus.join("B0.mmf")).unwrap();
        let events = parse_smaf(&b0);
        let first = events
            .iter()
            .find_map(|(time, event)| match event {
                SmafEvent::MaFmNote(note) => Some((*time, note)),
                _ => None,
            })
            .expect("B0 must route its authored FM voice internally");
        assert_eq!(first.0, 0);
        assert_eq!(first.1.channel, 1);
        assert_eq!(first.1.note, 57);
        assert_eq!(first.1.velocity, 85);
        assert_eq!(first.1.gate_ms, 744);
        assert_eq!(first.1.patch.algorithm, 5);
        assert_eq!(first.1.patch.operator_count, 4);

        // B0 has 13 Mtsu SysEx messages: 11 are custom voice definitions and
        // only the two device setup messages belong at the generic MIDI edge.
        let midi_sysex = events
            .iter()
            .filter(|(_, event)| matches!(event, SmafEvent::MidiSysEx(_)))
            .count();
        assert_eq!(midi_sysex, 2);
    }

    #[test]
    fn zic2_b7_bird_sample_is_decoded_independently_of_midi_soundfont() {
        use std::{fs, path::PathBuf};

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test_data/zic2_bgm/B7.mmf");
        let data = fs::read(path).unwrap();
        let events = parse_smaf(&data);
        let (time, channels, sampling_rate, pcm, dynamics) = events
            .iter()
            .find_map(|(time, event)| match event {
                SmafEvent::Wave {
                    channels,
                    sampling_rate,
                    data,
                    dynamics,
                } => Some((*time, *channels, *sampling_rate, data, *dynamics)),
                _ => None,
            })
            .expect("B7 must contain its authored Mwa1 sample");

        assert_eq!(time, 0);
        assert_eq!(channels, 1);
        assert_eq!(sampling_rate, 8000);
        assert_eq!(dynamics.velocity, 127);
        assert_eq!(dynamics.volume, 100);
        assert_eq!(dynamics.expression, 127);
        assert_eq!(dynamics.pan, None);
        // GateTime is 930 * 4 ms = 3720 ms; the physical Mwa1 payload is
        // slightly shorter, so end-of-payload is authoritative here.
        assert_eq!(pcm.len(), 29_756);
        assert_eq!(
            &pcm[..32],
            &[
                15, 0, -15, 94, 16, -157, 70, 40, -41, 31, -33, -130, 132, 8, -104, -4,
                26, -1, 71, 7, -129, 28, 89, -41, 42, 27, -115, 56, 169, -135, -87, 131,
            ]
        );
    }

}
