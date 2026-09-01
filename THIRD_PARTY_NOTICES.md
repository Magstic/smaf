# Third-party behavioral references

This project does not vendor Yamaha firmware, handset sample ROMs, proprietary
SDK code, or code from the LGT `liblgt_system.so` binary.

The MA-3/MA-5 FM renderer was independently implemented from SMAF data observed
in the supplied corpus and public format documentation. During validation of
envelope, operator and output-stage behavior, the following maintained open
implementation was also used as a behavioral reference:

- `akustikrausch/yamaha-smaf-player`
  - https://github.com/akustikrausch/yamaha-smaf-player
  - License: Apache License 2.0

No Yamaha sample ROM is included. The external project likewise documents that
the MA handset preset/sample ROM is unavailable; therefore file-defined PCM-ROM
voices cannot be reconstructed bit-exactly from an MMF that contains only ROM
voice metadata. In this project those notes remain at the explicitly documented
SoundFont/MIDI approximation boundary.
