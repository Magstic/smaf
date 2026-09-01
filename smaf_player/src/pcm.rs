use alloc::vec::Vec;
use smaf::BaseBit;

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum PcmEncoding {
    TwosComplement,
    OffsetBinary,
}

fn bits_per_sample(base_bit: BaseBit) -> u8 {
    match base_bit {
        BaseBit::Bit4 => 4,
        BaseBit::Bit8 => 8,
        BaseBit::Bit12 => 12,
        BaseBit::Bit16 => 16,
    }
}

fn read_be_bits(data: &[u8], bit_offset: usize, width: u8) -> u16 {
    let mut value = 0u16;
    for bit in 0..width as usize {
        let absolute = bit_offset + bit;
        let byte = data[absolute >> 3];
        let shift = 7 - (absolute & 7);
        value = (value << 1) | (((byte >> shift) & 1) as u16);
    }
    value
}

fn normalize_sample(raw: u16, width: u8, encoding: PcmEncoding) -> i16 {
    let shift = 16 - width as u32;
    let centered = match encoding {
        PcmEncoding::TwosComplement => {
            let sign_shift = 32 - width as u32;
            (((raw as u32) << sign_shift) as i32) >> sign_shift
        }
        PcmEncoding::OffsetBinary => raw as i32 - (1i32 << (width - 1)),
    };
    (centered << shift) as i16
}

/// Decode SMAF linear PCM into interleaved signed 16-bit PCM.
///
/// SMAF is a big-endian bitstream format.  Samples are consumed MSB-first and
/// channel samples are interleaved.  4/12-bit samples are packed without byte
/// padding; an incomplete trailing sample/frame is ignored rather than reading
/// past the chunk boundary.
pub fn decode_pcm(data: &[u8], base_bit: BaseBit, encoding: PcmEncoding, channels: u8) -> Option<Vec<i16>> {
    if channels != 1 && channels != 2 {
        return None;
    }

    let width = bits_per_sample(base_bit);
    let total_samples = data.len().saturating_mul(8) / width as usize;
    let complete_samples = total_samples - total_samples % channels as usize;
    let mut result = Vec::with_capacity(complete_samples);

    for sample_index in 0..complete_samples {
        let raw = read_be_bits(data, sample_index * width as usize, width);
        result.push(normalize_sample(raw, width, encoding));
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::{decode_pcm, PcmEncoding};
    use smaf::BaseBit;

    #[test]
    fn decodes_signed_8_bit_pcm() {
        assert_eq!(
            decode_pcm(&[0x80, 0x00, 0x7f], BaseBit::Bit8, PcmEncoding::TwosComplement, 1).unwrap(),
            [i16::MIN, 0, 32512]
        );
    }

    #[test]
    fn decodes_offset_binary_8_bit_pcm() {
        assert_eq!(
            decode_pcm(&[0x00, 0x80, 0xff], BaseBit::Bit8, PcmEncoding::OffsetBinary, 1).unwrap(),
            [i16::MIN, 0, 32512]
        );
    }

    #[test]
    fn decodes_big_endian_signed_16_bit_pcm() {
        assert_eq!(
            decode_pcm(&[0x80, 0x00, 0x12, 0x34, 0x7f, 0xff], BaseBit::Bit16, PcmEncoding::TwosComplement, 1).unwrap(),
            [i16::MIN, 0x1234, i16::MAX]
        );
    }

    #[test]
    fn decodes_packed_12_bit_samples_msb_first() {
        // 0x800, 0x000, 0x7ff packed as: 80 00 00 7f f0.
        assert_eq!(
            decode_pcm(&[0x80, 0x00, 0x00, 0x7f, 0xf0], BaseBit::Bit12, PcmEncoding::TwosComplement, 1).unwrap(),
            [i16::MIN, 0, 32752]
        );
    }

    #[test]
    fn drops_incomplete_stereo_frame() {
        assert_eq!(
            decode_pcm(&[0x01, 0x02, 0x03], BaseBit::Bit8, PcmEncoding::TwosComplement, 2).unwrap(),
            [256, 512]
        );
    }
}
