use core::time::Duration;
use std::{env::args, fs};

use midir::{MidiOutput, MidiOutputConnection};
use rodio::{buffer::SamplesBuffer, OutputStream, OutputStreamHandle, Sink};
use tokio::time::sleep;

use smaf_player::{parse_smaf, SmafEvent};

#[tokio::main(flavor = "current_thread")]
pub async fn main() {
    let mut file = None;
    let mut loop_playback = false;
    for argument in args().skip(1) {
        if argument == "--loop" {
            loop_playback = true;
        } else if file.is_none() {
            file = Some(argument);
        } else {
            panic!("Unexpected argument: {argument}");
        }
    }

    let file = file.expect("No file given");
    let data = fs::read(file).expect("Failed to read file");
    let events = parse_smaf(&data);

    // Pure sampled SFX do not require a MIDI device. Keep MIDI optional so
    // waveform playback works on emulator/test hosts without a MIDI port.
    let mut midi_out = MidiOutput::new("smaf_cli").ok().and_then(|midi| {
        let port = midi.ports().into_iter().last()?;
        midi.connect(&port, "smaf_cli").ok()
    });

    let (_output_stream, stream_handle) = OutputStream::try_default().expect("No audio output device");

    loop {
        play_events(&events, &mut midi_out, &stream_handle).await;
        if !loop_playback {
            break;
        }
    }
}

async fn play_events(events: &[(usize, SmafEvent)], midi_out: &mut Option<MidiOutputConnection>, stream_handle: &OutputStreamHandle) {
    let mut now = 0;
    let mut audio_sinks = Vec::new();
    for (time, event) in events {
        sleep(Duration::from_millis(time.saturating_sub(now) as u64)).await;

        match event {
            SmafEvent::Wave {
                channels,
                sampling_rate,
                data,
                dynamics,
            } => {
                // Decode data stays unscaled in smaf_player; sequence velocity,
                // Volume, Expression and Pan are applied at the renderer edge.
                let rendered = smaf_renderer::render_wave_event(
                    *channels,
                    *sampling_rate,
                    data,
                    *dynamics,
                    *sampling_rate,
                );
                let sink = Sink::try_new(stream_handle).expect("Failed to create audio sink");
                sink.append(SamplesBuffer::new(
                    rendered.channels as _,
                    rendered.sampling_rate,
                    rendered.data,
                ));
                audio_sinks.push(sink);
            }
            SmafEvent::MaFmNote(note) => {
                // Custom MA-3/MA-5 FM definitions live in the MMF itself and
                // are independent of the external GM SoundFont. Render them
                // through the MA FM path rather than changing their timbre by
                // forwarding the program number to an unrelated SF2.
                let rendered = smaf_renderer::render_ma_fm_note(note, 44_100);
                let sink = Sink::try_new(stream_handle).expect("Failed to create FM audio sink");
                sink.append(SamplesBuffer::new(
                    rendered.channels as _,
                    rendered.sampling_rate,
                    rendered.data,
                ));
                audio_sinks.push(sink);
            }
            SmafEvent::MidiNoteOn { channel, note, velocity } => send_midi(midi_out, &[0x90 | *channel, *note, *velocity]),
            SmafEvent::MidiNoteOff { channel, note, velocity } => send_midi(midi_out, &[0x80 | *channel, *note, *velocity]),
            SmafEvent::MidiProgramChange { channel, program } => send_midi(midi_out, &[0xC0 | *channel, *program]),
            SmafEvent::MidiControlChange { channel, control, value } => send_midi(midi_out, &[0xB0 | *channel, *control, *value]),
            SmafEvent::MidiPitchBend { channel, value } => {
                send_midi(midi_out, &[0xE0 | *channel, (*value & 0x7f) as u8, ((*value >> 7) & 0x7f) as u8]);
            }
            SmafEvent::MidiSysEx(data) => send_midi(midi_out, data),
            SmafEvent::End => {}
        }

        now = *time;
    }

    // Keep every audio sink alive and wait for envelope release tails before a
    // loop restarts. Detaching sinks here reintroduces cross-loop overlap.
    for sink in audio_sinks {
        sink.sleep_until_end();
    }
}

fn send_midi(midi_out: &mut Option<MidiOutputConnection>, message: &[u8]) {
    if let Some(midi_out) = midi_out.as_mut() {
        let _ = midi_out.send(message);
    }
}
