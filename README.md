# Rally Audio Pipeline

Downloads rally video clips from YouTube, separates co-driver speech from engine/road noise, transcribes the speech, and produces synchronized spectrogram analysis figures.

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | **Edit this first.** Your personal settings (Box path, Whisper model). |
| `clips.csv` | List of YouTube clips to process with timestamps. |
| `download.py` | Main pipeline — runs all steps for every clip in clips.csv. |
| `speechbrain_split.py` | Separates speech and noise stems using DeepFilterNet. |
| `analyze.py` | Generates spectrogram figures and time-series analysis. |
| `requirements.txt` | Python package dependencies. |

---

## Setup

### 1. Install Conda (if you don't have it)
Download Miniconda from https://docs.conda.io/en/latest/miniconda.html

### 2. Create the environment
```bash
conda create -n rally-audio python=3.10 -y
conda activate rally-audio
```

### 3. Install system tools
```bash
conda install -c conda-forge yt-dlp ffmpeg -y
```

### 4. Install Python packages
```bash
pip install -r requirements.txt
```

> **Note on numpy:** DeepFilterNet requires numpy <2. If you see a numpy conflict, run:
> `pip install numpy==1.26.4`

### 5. Edit config.py
Open `config.py` and set `BOX_FOLDER` to your local Box sync path:

```python
# Mac
BOX_FOLDER = "/Users/yourname/Library/CloudStorage/Box-Box/GroupD"

# Windows
BOX_FOLDER = "C:/Users/yourname/Box/GroupD"
```

Set to `None` if you don't want Box export.

---

## Usage

### Step 1 — Fill in clips.csv
```
url,start,end
https://www.youtube.com/watch?v=5gJxWIybplc,0:30,1:45
https://www.youtube.com/watch?v=abc123,2:00,3:30
```
- `start` and `end` are optional — leave blank to download the full video
- Timestamps can be `mm:ss`, `hh:mm:ss`, or plain seconds
- Lines starting with `#` are ignored

The shared `clips.csv` lives in the Box folder — copy it locally before running.

### Step 2 — Run the pipeline
```bash
conda activate rally-audio
python download.py
```

This processes every clip in `clips.csv` and for each one:
1. Downloads the clip from YouTube at the specified timestamps
2. Extracts audio at 48kHz with loudness normalization
3. Separates speech and noise stems using DeepFilterNet
4. Transcribes the speech stem using Whisper
5. Copies outputs to your Box folder

### Step 3 — Generate analysis figures
```bash
python analyze.py --all
```

Or for a single clip:
```bash
python analyze.py raw/audio_1.wav stems/video_1/speech.wav stems/video_1/noise.wav
```

Figures are saved to `analysis/` and exported to Box automatically.

### One-off local MP4
If you already have a downloaded mp4:
```bash
ffmpeg -y -i your_file.mp4 -ar 48000 -ac 1 -af "loudnorm=I=-16:TP=-1.5:LRA=11" -c:a pcm_s16le raw/audio_1.wav
python speechbrain_split.py raw/audio_1.wav stems/video_1/
python analyze.py raw/audio_1.wav stems/video_1/speech.wav stems/video_1/noise.wav
```

---

## Outputs

```
local/
├── raw/
│   ├── video_1.mp4          # downloaded clip
│   └── audio_1.wav          # extracted audio (48kHz)
├── stems/
│   └── video_1/
│       ├── speech.wav       # co-driver voice (DeepFilterNet enhanced)
│       ├── noise.wav        # engine/road noise (speech suppressed)
│       └── speech_16k.wav   # downsampled for Whisper (16kHz)
├── transcript/
│   └── video_1/             # Whisper output (txt, json, srt, vtt, tsv)
└── analysis/
    └── video_1.png          # spectrogram + time-series figure

Box/GroupD/outputs/
└── video_1/
    ├── speech.wav
    ├── noise.wav
    ├── speech_16k.wav
    ├── *.txt / *.json / *.srt   # transcripts
    └── video_1.png              # analysis figure
```

---

## Reading the Analysis Figure

The figure has four panels, all sharing a time axis:

**Panels 1–3 — Spectrograms** (frequency vs time, brightness = energy)
- Dashed lines mark the human voice band (300Hz – 3.4kHz)
- **Original** — raw mixed audio before separation
- **Speech stem** — co-driver voice isolated; bright energy should be inside the dashed lines
- **Noise stem** — engine/road noise; bright energy should be outside the dashed lines

**Panel 4 — Time series**
- Green area: speech RMS — how loud the co-driver is at each moment
- Orange line: noise spectral centroid — rises with engine RPM, falls during deceleration

---

## Troubleshooting

**Wrong conda environment error**
```bash
conda activate rally-audio
```

**`No module named 'df'` (DeepFilterNet)**
```bash
pip install deepfilternet
```

**DeepFilterNet model download stalls**
The model (~60MB) downloads from HuggingFace on first run. Check your network and retry.

**yt-dlp fails to download**
```bash
pip install -U yt-dlp
```

**Whisper is slow**
Switch to a smaller model in `config.py`: `WHISPER_MODEL = "tiny"`

---

## A note on separation quality

The speech and noise stems will have some overlap in the 300–3400Hz band. This is a fundamental physical limitation of single-channel audio — engine harmonics and human voice occupy the same frequencies. The pipeline produces the best separation possible from a single microphone recording, which is sufficient for amplitude and frequency analysis.