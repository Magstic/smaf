use alloc::vec::Vec;

// Yamaha 4-bit ADPCM (the codec FFmpeg exposes as ADPCM_YAMAHA).
//
// Each nibble carries a sign bit and a 3-bit magnitude.  The predictor starts
// at zero and the adaptive step starts at 127.  The scale table is specified in
// Q8; do not replace it with a /64 approximation, because that changes the
// state after every nibble and quickly destroys the waveform.
const STEP_SCALE_Q8: [u16; 8] = [230, 230, 230, 230, 307, 409, 512, 614];
const STEP_MIN: u32 = 127;
const STEP_MAX: u32 = 24576;

#[derive(Copy, Clone)]
struct DecodeContext {
    predictor: i32,
    step: u32,
}

impl DecodeContext {
    const fn new() -> Self {
        Self { predictor: 0, step: STEP_MIN }
    }

    fn decode_nibble(&mut self, nibble: u8) -> i16 {
        debug_assert!(nibble < 16);

        let magnitude = (nibble & 0x07) as u32;
        let difference = ((magnitude * 2 + 1) * self.step) >> 3;
        if nibble & 0x08 != 0 {
            self.predictor -= difference as i32;
        } else {
            self.predictor += difference as i32;
        }
        self.predictor = self.predictor.clamp(i16::MIN as i32, i16::MAX as i32);

        self.step = ((self.step * STEP_SCALE_Q8[magnitude as usize] as u32) >> 8).clamp(STEP_MIN, STEP_MAX);
        self.predictor as i16
    }
}

/// Decode Yamaha 4-bit ADPCM into interleaved signed 16-bit PCM.
///
/// SMAF uses the same byte ordering as FFmpeg's ADPCM_YAMAHA decoder:
///
/// * mono: low nibble first, then high nibble, both through one state machine;
/// * stereo: low nibble is left and high nibble is right, with independent
///   predictor/step state for each channel.
pub fn decode_yamaha_adpcm(data: &[u8], channels: u8) -> Option<Vec<i16>> {
    match channels {
        1 => {
            let mut result = Vec::with_capacity(data.len().saturating_mul(2));
            let mut context = DecodeContext::new();
            for &byte in data {
                result.push(context.decode_nibble(byte & 0x0f));
                result.push(context.decode_nibble(byte >> 4));
            }
            Some(result)
        }
        2 => {
            let mut result = Vec::with_capacity(data.len().saturating_mul(2));
            let mut left = DecodeContext::new();
            let mut right = DecodeContext::new();
            for &byte in data {
                result.push(left.decode_nibble(byte & 0x0f));
                result.push(right.decode_nibble(byte >> 4));
            }
            Some(result)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::decode_yamaha_adpcm;

    #[test]
    fn decodes_low_nibble_before_high_nibble() {
        // 0x10 must decode nibble 0 first and nibble 1 second.  Reversing the
        // nibble order gives a different second-order predictor state.
        assert_eq!(decode_yamaha_adpcm(&[0x10], 1).unwrap(), [15, 62]);
    }

    #[test]
    fn stereo_uses_independent_channel_state() {
        // The first byte initializes L with nibble 0 and R with nibble 1.  The
        // second byte must continue each channel's own predictor history.
        assert_eq!(decode_yamaha_adpcm(&[0x10, 0x32], 2).unwrap(), [15, 47, 94, 158]);
    }

    #[test]
    fn rejects_unsupported_channel_count() {
        assert!(decode_yamaha_adpcm(&[0x00], 0).is_none());
        assert!(decode_yamaha_adpcm(&[0x00], 3).is_none());
    }
}
