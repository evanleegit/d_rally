"""
download.py - Rally Race Audio Analysis Pipeline
-------------------------------------------------
Reads clips.csv (or a CSV you specify) and processes each clip end-to-end:
  1. Download clip at specified timestamps (yt-dlp)
  2. Extract audio at 48kHz with loudness normalization (ffmpeg)
  3. Separate speech vs car noise (split.py → DeepFilterNet)
  4. Transcribe speech stem (Whisper)
  5. Export outputs to Box (if configured in config.py)

CSV format:
    url,start,end
    https://www.youtube.com/watch?v=5gJxWIybplc,0:30,1:45
    https://www.youtube.com/watch?v=abc123,2:00,3:30

Timestamps accept: seconds (90), mm:ss (1:30), or hh:mm:ss (0:01:30)
Start/end are optional — leave blank to download the full video.

Usage:
    python download.py              # reads clips.csv in current directory
    python download.py myclips.csv  # reads a specific csv
"""

import csv
import shutil
import subprocess
import sys
import os
from pathlib import Path

# ── Load config ───────────────────────────────────────────────────────────────
try:
    from config import CONDA_ENV, WHISPER_MODEL, WHISPER_DEVICE, ENABLE_WHISPER, get_box_path
except ImportError:
    print("[ERROR] config.py not found. Make sure it is in the same folder as download.py.")
    sys.exit(1)

# ── Environment check ─────────────────────────────────────────────────────────
current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
if current_env != CONDA_ENV:
    print(f"""
[ERROR] Wrong conda environment
  Current : {current_env or '(none)'}
  Required: {CONDA_ENV}

Fix:
  conda activate {CONDA_ENV}
  python download.py
""")
    sys.exit(1)

# ── Executable check ──────────────────────────────────────────────────────────
REQUIRED_EXECUTABLES = {
    "yt-dlp":  "conda install -c conda-forge yt-dlp",
    "ffmpeg":  "conda install -c conda-forge ffmpeg",
    "whisper": "pip install openai-whisper",
}

def check_executables():
    missing = []
    for exe, install_cmd in REQUIRED_EXECUTABLES.items():
        if shutil.which(exe) is None:
            missing.append((exe, install_cmd))
    if missing:
        print("\n[ERROR] Missing required tools:\n")
        for exe, cmd in missing:
            print(f"  {exe:10s} → install with:  {cmd}")
        sys.exit(1)

check_executables()

# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list, label: str = ""):
    tag = f"[{label}] " if label else ""
    print(f"\n{tag}Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n{tag}[ERROR] Command failed with exit code {result.returncode}")
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}")


def conda_python() -> str:
    """Resolve the Python interpreter inside the active conda environment."""
    conda_root = os.environ.get("CONDA_PREFIX", "")
    if conda_root:
        candidate = Path(conda_root) / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env:
        try:
            result = subprocess.run(
                ["conda", "run", "-n", conda_env, "which", "python"],
                capture_output=True, text=True
            )
            path = result.stdout.strip()
            if path and Path(path).exists():
                return path
        except Exception:
            pass
    print("[WARN] Could not resolve conda Python via CONDA_PREFIX; using sys.executable")
    return sys.executable


def normalise_timestamp(ts: str) -> str:
    """Convert any timestamp format to hh:mm:ss for yt-dlp."""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 1:
        total = int(parts[0])
        h, rem = divmod(total, 3600)
        m, s   = divmod(rem, 60)
    elif len(parts) == 2:
        h, m, s = 0, int(parts[0]), int(parts[1])
    else:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_csv(csv_path: str) -> list:
    clips = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader, start=2):
            url = (row.get("url") or "").strip()

            if not url or url.startswith("#"):
                continue

            start_raw = (row.get("start") or "").strip()
            end_raw   = (row.get("end") or "").strip()

            clips.append({
                "url": url,
                "start": normalise_timestamp(start_raw) if start_raw else None,
                "end": normalise_timestamp(end_raw) if end_raw else None,
            })

    return clips


def export_to_box(box_root: Path, clip_label: str, files: list):
    """Copy finished output files into box_root/outputs/clip_label/."""
    dest = box_root / "outputs" / clip_label
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        if Path(f).exists():
            shutil.copy2(str(f), str(dest / Path(f).name))
            print(f"[Box]  → {dest / Path(f).name}")
        else:
            print(f"[Box]  Skipping (not found): {f}")


# ── Directory setup ───────────────────────────────────────────────────────────
for d in ["raw", "transcript", "stems"]:
    Path(d).mkdir(exist_ok=True)

# ── Resolve CSV ───────────────────────────────────────────────────────────────
csv_path = sys.argv[1] if len(sys.argv) > 1 else "clips.csv"
if not Path(csv_path).exists():
    print(f"[ERROR] CSV file not found: {csv_path}")
    print("  Create a clips.csv with columns: url, start, end")
    print("  Example: https://youtube.com/watch?v=abc,0:30,1:45")
    sys.exit(1)

clips = load_csv(csv_path)
if not clips:
    print(f"[ERROR] No valid clips found in {csv_path}")
    sys.exit(1)

# ── Box path ──────────────────────────────────────────────────────────────────
box_path = get_box_path()
if box_path:
    print(f"\n[Box]  Outputs will be exported to: {box_path / 'outputs'}")
else:
    print("\n[Box]  No Box folder found — outputs will be local only.")

# ── Resolve split script ──────────────────────────────────────────────────────
split_script = Path(__file__).parent / "split.py"
if not split_script.exists():
    print(f"[ERROR] split.py not found at {split_script}")
    sys.exit(1)

print(f"\nLoaded {len(clips)} clip(s) from {csv_path}")

# ── Main loop ─────────────────────────────────────────────────────────────────
for idx, clip in enumerate(clips, start=1):
    url        = clip["url"]
    start      = clip["start"]
    end        = clip["end"]
    time_label = f"{start}-{end}" if start and end else "full"
    clip_label = f"video_{idx}"

    print(f"\n{'='*60}")
    print(f"  Clip {idx}/{len(clips)}  [{time_label}]")
    print(f"  URL: {url}")
    print(f"{'='*60}")

    raw_dir = Path(f"raw/video_{idx}")
    raw_dir.mkdir(parents=True, exist_ok=True)

    video_file = raw_dir / "video.%(ext)s"   # ← key change
    audio_file = raw_dir / "audio.wav"

    stems_dir  = Path(f"stems/video_{idx}")
    stems_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download ──────────────────────────────────────────────────
    print(f"\n[Step 1/4] Downloading clip [{time_label}]...")
    yt_cmd = [
        "yt-dlp",
        "-f", "bestvideo+bestaudio",
        "-o", str(video_file),
        "--merge-output-format", "mp4",
        "--no-playlist",
    ]
    if start and end:
        yt_cmd += ["--download-sections", f"*{start}-{end}"]
    yt_cmd.append(url)
    run(yt_cmd, "yt-dlp")

    actual_video = None
    for ext in ["mp4", "mkv", "webm"]:
        candidate = raw_dir / f"video.{ext}"
        if candidate.exists():
            actual_video = str(candidate)
            break
    if not actual_video:
        print(f"[ERROR] Could not find downloaded video file for clip {idx}")
        continue

    # ── Step 2: Extract audio ─────────────────────────────────────────────
    print("\n[Step 2/4] Extracting audio (mono, 48kHz, loudness normalized)...")
    run([
        "ffmpeg", "-y",
        "-i", actual_video,
        "-ar", "48000",
        "-ac", "1",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "pcm_s16le",
        audio_file
    ], "ffmpeg")

    # ── Step 3: Separate speech vs noise ──────────────────────────────────
    print("\n[Step 3/4] Separating speech and car noise...")
    run([
        conda_python(),
        str(split_script),
        audio_file,
        str(stems_dir),
    ], "separation")

    # ── Step 4: Transcribe with Whisper ───────────────────────────────────
    WHISPER_MODEL = "small"
    WHISPER_DEVICE = "cpu"

    speech_stem = stems_dir / "speech.wav"
    speech_16k  = stems_dir / "speech_16k.wav"

    if ENABLE_WHISPER:
        print("\n[Step 4/4] Downsampling speech stem and transcribing with Whisper...")
        run([
            "ffmpeg", "-y",
            "-i", str(speech_stem),
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(speech_16k)
        ], "ffmpeg")

        transcript_dir = Path("transcript") / clip_label
        transcript_dir.mkdir(parents=True, exist_ok=True)
        run([
            "whisper", str(speech_16k),
            "--model",           WHISPER_MODEL,
            "--word_timestamps", "True",
            "--output_format",   "all",
            "--output_dir",      str(transcript_dir),
            "--device",          WHISPER_DEVICE,
        ], "whisper")
        print(f"  Transcripts saved to: {transcript_dir}/")
    else:
        print("\n[Step 4/4] Whisper transcription skipped (ENABLE_WHISPER=False)")

    # ── Step 5: Export to Box ─────────────────────────────────────────────
    if box_path:
        print(f"\n[Step 5/4] Exporting to Box...")
        export_files = [
            stems_dir / "speech.wav",
            stems_dir / "noise.wav",
            speech_16k,
        ]
        if ENABLE_WHISPER:
            transcript_dir = Path("transcript") / clip_label
            for ext in ["txt", "json", "srt", "vtt", "tsv"]:
                export_files.extend(transcript_dir.glob(f"*.{ext}"))
        export_to_box(box_path, clip_label, export_files)

    print(f"\n  ✓ Clip {idx} done  [{time_label}]")
    print(f"    Audio   : {audio_file}")
    print(f"    Speech  : {stems_dir}/speech.wav")
    print(f"    Noise   : {stems_dir}/noise.wav")
    if ENABLE_WHISPER:
        print(f"    Transcript : transcript/{clip_label}/")
    if box_path:
        print(f"    Box     : {box_path / 'outputs' / clip_label}/")

print(f"\n{'='*60}")
print(f"  All {len(clips)} clip(s) processed successfully!")
if box_path:
    print(f"  Outputs in Box: {box_path / 'outputs'}")
print(f"{'='*60}\n")