# Beat tracking

The **`beat-track`** stage finds the beats of music in a PCM stream and
hands out where they fall: in seconds, in samples or in onset-envelope
frames. Beats are packed `group` to a FlexData object, ready for a light
show, a video cut on the beat or a tempo readout.

The tracker is [librosa](https://librosa.org)'s `beat_track` (librosa
0.11), the dynamic-programming beat tracker of Ellis, *Beat Tracking by
Dynamic Programming* (JNMR 2007). vpipe reproduces librosa's arithmetic
step by step, in plain C++ on the CPU with no Python. Given the same
samples, the whole-stream mode gives librosa's tempo and its beats,
frame for frame.

## What you need

| | |
|---|---|
| **Machine** | Any Apple Silicon Mac. |
| **Memory** | A CPU stage. `stream` holds about 2 MB (a window of the spectrogram and the tracker's recent history), at most about 6 MB through a long silence, plus its beats so far, about 0.25 MB an hour; `whole` holds every frame until the end, about 550 bytes a frame (1.4 MB a minute of audio). A 3½-minute song takes about 0.15 s. |
| **Model** | None. |
| **FFmpeg** | Only to convert a stream that is not already at 22050 Hz (libavfilter's `aresample`). |

## Run it

[`beat-track.vpipeline`](pipelines/beat-track.vpipeline) reads a file, tracks
its beats and writes one JSON object per four beats to
`song-beats.jsonl`:

```
load-audio → audio-to-pcm (22050 Hz, stereo) → beat-track → save-text (format json)
```

```sh
vpipe --launch docs/pipelines/beat-track.vpipeline \
      --stage-cfg load-audio::input_url=song.m4a \
      --stage-cfg save-beats::path=song-beats.jsonl
```

Each line looks like this:

```json
{"beats":[0.534,1.045,1.556,2.067],"units":"time","index":0,
 "tempo":117.45,"sample_rate":22050,"hop_length":512,
 "analysis_rate":22050,"timestamp_us":0}
```

| field | |
|---|---|
| `beats` | The beats in `units`. With `time`, seconds from the stream's first sample. With `samples`, the sample at the INPUT's rate. With `frames`, onset-envelope frames of `hop_length` samples at the analysis rate. |
| `index` | The ordinal of `beats[0]` in the stream, so the groups join up. |
| `tempo` | The tempo, in bpm, the beats were placed under: the estimate at the time in `stream` mode, the whole track's in `whole`. |
| `sample_rate` | The input's rate. `analysis_rate` is the rate the tracker heard. |
| `channel` | Only with `channels: each` or a channel index: which channel. |
| `timestamp_us` | The UTC of the stream's first sample, when the input carries one (live capture). |

The stage takes f32 PCM, `[N]` mono or planar `[C, N]`, with `sample_rate`
on the sideband: what `audio-to-pcm`, `audio-capture` and
`audio-temporal-resample` emit. Input beats of any length are one stream
until the end of stream. A relaunch starts a new one.

## Two modes

**`stream`** (the default) hands beats out while the sound goes on. Three
of librosa's choices need the whole track: the dB ceiling, the tempo and
the envelope's deviation. A stream takes each over what it has heard so
far. A beat is handed out once **every path the tracker could still
choose runs through it**, so nothing handed out is ever revised.

- **The first beats** come out once 8 seconds have been heard past the
  first onset. That is the tempogram's window; the tempo needs it.
- **After that** most beats are certain 2–4 seconds after they are
  heard. In a passage where two phases of the beat stay plausible,
  certainty can take 10–20 seconds.
- **`max_latency`** (default 4 s) bounds the wait. A beat that old is
  handed out on the best path so far. If a later decision disagrees, the
  stream carries on from what was said rather than revising it.
- A stream that ends before its first decision is decided exactly as
  `whole` would.

**`whole`** waits for the end of the stream and hands out librosa's
answer for all of it.

How close the stream lands to librosa, measured on two whole songs
(3:34 and 4:45) and a minute excerpt, scored away from the first and last
2.5 s:

| `max_latency` | beats within one frame (23 ms) of librosa's | 95% of beats out within |
|---|---|---|
| 0 (wait for certainty) | 98.8–100% | 4.5–11.5 s |
| 4 s (default) | 98.1–100% | 4.2 s |
| 2 s | 96.9–100% | 2.2 s |
| 1 s | 95.5–100% | 1.2 s |

Near either end, the stream and librosa can keep or drop a weak beat
differently. librosa sets its `trim` threshold from the whole track's
beats; a stream can only use the beats it has so far. After a long
silence the difference can be larger: the deviation the envelope is
divided by is still growing out of the silence's zeros, so the first
seconds of music score high, and so does the threshold. Four minutes
of silence then twenty seconds of clicks lost the last 4.5 seconds of
beats to the trailing trim, where librosa keeps them.

## Configuration

| key | default | |
|---|---|---|
| `mode` | `stream` | `stream` or `whole` (above). |
| `group` | 1 | Beats per object. The last object of a stream holds what is left. |
| `units` | `time` | `time`, `samples` or `frames`. |
| `channels` | `mix` | `mix`: one tracker on the channels' mean, as `librosa.load` hears a file. `each`: a tracker per channel. Or a channel index. |
| `sample_rate` | 22050 | The rate the tracker hears, `librosa.load`'s. Another input rate is converted. 0 tracks at the input's own rate, as `beat_track(y, sr=that rate)` does. |
| `hop_length` | 512 | The onset envelope's hop (23 ms at 22050 Hz). At most 1000 envelope frames a second. |
| `start_bpm` | 120 | The centre of the tempo prior. |
| `tightness` | 100 | How closely beats keep to the tempo; lower follows the onsets more. |
| `trim` | true | Drop weak leading and trailing beats. |
| `bpm` | 0 | Above 0: track at this tempo instead of estimating one. |
| `max_latency` | 4 | `stream`: the longest a beat waits, in seconds; 0 waits for certainty. |

## What it does

These are librosa's steps, with its defaults:

1. **Onset strength.** A centred STFT (2048 points, periodic Hann). The
   power goes through a 128-band Slaney mel filterbank to half the
   sample rate, then into dB with an 80 dB floor below the loudest bin.
   Each band's rise from the previous frame is taken (falls clipped to
   zero), and the median over the bands is the envelope.
2. **Tempo.** The envelope's autocorrelation in 8-second Hann windows,
   averaged. The tempo is the lag of greatest `log1p(1e6 · ac)` under a
   log-normal prior about `start_bpm`, one octave wide, with tempi above
   320 bpm excluded.
3. **Beats.** The envelope over its deviation is smoothed by a Gaussian a
   32nd of a period wide. A dynamic program takes each frame's best
   predecessor between half and two periods back, penalised by
   `tightness · log(gap / period)²`. No chain may start below 1% of the
   strongest frame. The last beat is the last local maximum of the
   cumulative score that is at least half their median, and the path is
   traced back from it. The weak beats before the first, and after the
   last, frame stronger than half the beats' smoothed RMS are dropped.

## How it was checked

The reference is librosa 0.11's beat tracker **restated** in numpy
(`tools/dump_beat_track_golden.py`). librosa's source was read; librosa
itself was not run. Where librosa runs numpy, the restatement runs the
same numpy calls, so those values are librosa's to the bit. Where
librosa runs numba, its loops are written out in float64, as the kernels
numba picks there run.

On two whole songs, a minute and six seconds excerpted from one of them,
and click tracks at 128 and 93 bpm:

- **Fed librosa's onset envelope:** the tempo, the envelope's deviation,
  every local and cumulative score and every beat are **identical**.
- **From the samples:** the envelope is within float32 rounding (rel-L2
  ~1e-7; librosa's mel product is a BLAS call whose order cannot be
  reproduced). The tempo and every beat are **identical**.
- **Stream mode** is the same however the input is cut, from one sample
  at a time to the whole track at once. A stream that ends undecided is
  the whole track's answer. Twenty minutes of clicks put 2400 beats on
  2400 clicks while holding at most 4786 frames (110 s) of history.
- **Silences.** A stream that opens with silence holds nothing for it:
  three hours of silence before a song peaked at 29 MB and gave, bit
  for bit, the beats of a tracker that forgets nothing (529 MB).

The tests are `beat_track.*` and `beat_track_stage.*`, with the goldens
under `VPIPE_BEAT_TRACK_GOLDEN`.

Two differences from `librosa.load` + `beat_track` remain:

- **The conversion to 22050 Hz** is libavfilter's resampler where
  librosa's is soxr's. On a 48 kHz song converted inside the stage,
  every beat of the whole-stream mode was within a frame of librosa's.
- **Very long silences.** In a silence between two passages the
  tracker's candidate beats run side by side until the sound after it
  decides between them, which librosa, with the whole track, always
  knows. A stream waits 2^16 envelope frames (about 25 minutes at the
  default 43 a second) and then commits the best path so far. A beat
  held that long for want of a strong frame after it is dropped, as the
  trailing trim would drop it had the stream ended there. So a stream
  holds a bounded window however long it is silent. With 30 minutes of
  silence between two songs, the beats of both songs were those of a
  tracker that waits indefinitely; only the phase of the first beats
  placed in the silence differed.
