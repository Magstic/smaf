use alloc::vec::Vec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaVoiceKey {
    pub bank_msb: u8,
    pub bank_lsb: u8,
    pub program: u8,
    pub drum_note: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaFmOperatorPatch {
    pub sr: u8,
    pub ksr: bool,
    pub xof: bool,
    pub rr: u8,
    pub dr: u8,
    pub ar: u8,
    pub sl: u8,
    pub tl: u8,
    pub ksl: u8,
    pub am: bool,
    pub dam: u8,
    pub vib: bool,
    pub dvb: u8,
    pub multi: u8,
    pub dt: u8,
    pub wave: u8,
    pub feedback: u8,
}

impl Default for MaFmOperatorPatch {
    fn default() -> Self {
        Self {
            sr: 0,
            ksr: false,
            xof: false,
            rr: 0,
            dr: 0,
            ar: 15,
            sl: 0,
            tl: 63,
            ksl: 0,
            am: false,
            dam: 0,
            vib: false,
            dvb: 0,
            multi: 1,
            dt: 3,
            wave: 0,
            feedback: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaFmPatch {
    pub algorithm: u8,
    pub operator_count: u8,
    pub note_shift: i8,
    /// -1.0 = left, 0 = centre, +1.0 = right.
    pub default_pan: f32,
    pub lfo: u8,
    pub operators: [MaFmOperatorPatch; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaPcmRomVoice {
    pub sampling_rate: u16,
    pub total_level: u8,
    pub sr: u8,
    pub rr: u8,
    pub dr: u8,
    pub ar: u8,
    pub sl: u8,
    pub loop_point: u16,
    pub end_point: u16,
    pub looping: bool,
    pub wave_id: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaFmNote {
    pub channel: u8,
    pub note: u8,
    pub velocity: u8,
    pub gate_ms: u32,
    pub volume: u8,
    pub expression: u8,
    /// Explicit channel pan. None means use the custom voice's authored default pan.
    pub pan: Option<u8>,
    /// MIDI-compatible 14-bit bend value, centre = 8192, default range ±2 semitones.
    pub pitch_bend: u16,
    pub patch: MaFmPatch,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaCustomVoice {
    Fm { key: MaVoiceKey, patch: MaFmPatch },
    PcmRom { key: MaVoiceKey, params: MaPcmRomVoice },
}

impl MaCustomVoice {
    pub fn key(&self) -> MaVoiceKey {
        match *self {
            Self::Fm { key, .. } | Self::PcmRom { key, .. } => key,
        }
    }
}

fn bo_to_semitones(value: u8) -> i8 {
    match value & 3 {
        0 => 12,
        1 => 0,
        2 => -12,
        _ => -24,
    }
}

fn panpot_to_pan(value: u8) -> f32 {
    let value = value & 31;
    if value == 15 {
        0.0
    } else if value < 15 {
        -((15 - value) as f32 / 15.0)
    } else {
        (value - 15) as f32 / 16.0
    }
}

fn parse_vm35_body(body: &[u8], operator_count: usize) -> Option<MaFmPatch> {
    if body.len() < 3 {
        return None;
    }
    let algorithm = body[2] & 7;
    let mut patch = MaFmPatch {
        algorithm,
        operator_count: operator_count.clamp(1, 4) as u8,
        note_shift: bo_to_semitones(body[1] & 3),
        default_pan: if body[2] & 0x20 != 0 {
            panpot_to_pan((body[1] >> 3) & 31)
        } else {
            0.0
        },
        lfo: (body[2] >> 6) & 3,
        operators: [MaFmOperatorPatch::default(); 4],
    };

    let mut offset = 3usize;
    for operator in patch.operators.iter_mut().take(patch.operator_count as usize) {
        let raw = body.get(offset..offset + 7)?;
        operator.sr = (raw[0] >> 4) & 15;
        operator.ksr = raw[0] & 1 != 0;
        operator.xof = raw[0] & 8 != 0;
        operator.rr = (raw[1] >> 4) & 15;
        operator.dr = raw[1] & 15;
        operator.ar = (raw[2] >> 4) & 15;
        operator.sl = raw[2] & 15;
        operator.tl = (raw[3] >> 2) & 63;
        operator.ksl = raw[3] & 3;
        operator.am = raw[4] & 0x10 != 0;
        operator.dam = (raw[4] >> 5) & 3;
        operator.vib = raw[4] & 1 != 0;
        operator.dvb = (raw[4] >> 1) & 3;
        operator.multi = (raw[5] >> 4) & 15;
        operator.dt = raw[5] & 7;
        operator.wave = (raw[6] >> 3) & 31;
        operator.feedback = raw[6] & 7;
        offset += 7;
    }
    Some(patch)
}

fn parse_ma3_packed_fm(body: &[u8]) -> Option<MaFmPatch> {
    if body.len() < 4 {
        return None;
    }
    let algorithm = body[3] & 7;
    let operator_count = if algorithm <= 1 { 2 } else { 4 };

    let mut raw = [0u8; 36];
    let copied = body.len().min(raw.len());
    raw[..copied].copy_from_slice(&body[..copied]);

    // MA-3 VM3Exclusive stores the MSBs of several VM35 fields in carrier
    // bits. Restore them before removing the carrier bytes.
    raw[2] |= (raw[0] << 2) & 0x80;
    raw[3] |= (raw[0] << 3) & 0x80;
    for op in 0..4usize {
        raw[4 + op * 8] |= (raw[op * 8] << 4) & 0x80;
        raw[5 + op * 8] |= (raw[op * 8] << 5) & 0x80;
        raw[6 + op * 8] |= (raw[op * 8] << 6) & 0x80;
        raw[7 + op * 8] |= (raw[op * 8] << 7) & 0x80;
        raw[10 + op * 8] |= (raw[8 + op * 8] << 2) & 0x80;
        raw[11 + op * 8] |= (raw[8 + op * 8] << 3) & 0x80;
    }

    let mut fixed = Vec::with_capacity(3 + 7 * 4);
    fixed.extend_from_slice(&raw[1..4]);
    for op in 0..4usize {
        fixed.extend_from_slice(&raw[4 + op * 8..8 + op * 8]);
        fixed.extend_from_slice(&raw[9 + op * 8..12 + op * 8]);
    }
    parse_vm35_body(&fixed, operator_count)
}

pub fn parse_ma_voice_exclusive(data: &[u8]) -> Option<MaCustomVoice> {
    let data = data.strip_suffix(&[0xf7]).unwrap_or(data);
    if data.len() < 11
        || data[0] != 0x43
        || data[1] != 0x79
        || !matches!(data[2], 0x06 | 0x07)
        || data[3] != 0x7f
        || data[4] != 0x01
    {
        return None;
    }

    let key = MaVoiceKey {
        bank_msb: data[5] & 0x7f,
        bank_lsb: data[6] & 0x7f,
        program: data[7] & 0x7f,
        drum_note: data[8] & 0x7f,
    };
    let voice_type = data[9];
    let body = &data[10..];

    if voice_type != 0 {
        if body.len() < 16 {
            return None;
        }
        let sampling_rate = u16::from_be_bytes([body[0], body[1]]);
        let params = MaPcmRomVoice {
            sampling_rate: if (2000..=48000).contains(&sampling_rate) {
                sampling_rate
            } else {
                8000
            },
            total_level: (body[7] >> 2) & 63,
            sr: (body[4] >> 4) & 15,
            rr: (body[5] >> 4) & 15,
            dr: body[5] & 15,
            ar: (body[6] >> 4) & 15,
            sl: body[6] & 15,
            loop_point: u16::from_be_bytes([body[11], body[12]]),
            end_point: u16::from_be_bytes([body[13], body[14]]),
            looping: body[15] & 0x80 != 0,
            wave_id: body[15] & 0x7f,
        };
        return Some(MaCustomVoice::PcmRom { key, params });
    }

    let patch = if data[2] == 0x06 {
        parse_ma3_packed_fm(body)?
    } else {
        let algorithm = *body.get(2)? & 7;
        parse_vm35_body(body, if algorithm <= 1 { 2 } else { 4 })?
    };
    Some(MaCustomVoice::Fm { key, patch })
}

pub fn parse_setup_custom_voices(data: &[u8]) -> Vec<MaCustomVoice> {
    let mut result = Vec::new();
    let mut offset = 0usize;
    while offset < data.len() {
        if data[offset] != 0xf0 {
            break;
        }
        offset += 1;
        let Some((length, next_offset)) = read_vlq(data, offset) else {
            break;
        };
        offset = next_offset;
        let Some(message) = data.get(offset..offset.saturating_add(length)) else {
            break;
        };
        if let Some(voice) = parse_ma_voice_exclusive(message) {
            result.push(voice);
        }
        offset += length;
    }
    result
}

fn read_vlq(data: &[u8], mut offset: usize) -> Option<(usize, usize)> {
    let mut value = 0usize;
    for _ in 0..5 {
        let byte = *data.get(offset)?;
        offset += 1;
        value = value.checked_shl(7)? | usize::from(byte & 0x7f);
        if byte & 0x80 == 0 {
            return Some((value, offset));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_voice_yamaha_reset() {
        assert!(parse_ma_voice_exclusive(&[0x43, 0x79, 0x06, 0x7f, 0x7f, 0xf7]).is_none());
    }
}
