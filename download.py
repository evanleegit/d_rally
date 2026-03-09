"""
download.py - Rally Race Audio Analysis Pipeline
-------------------------------------------------
Steps per video:
  1. Download video (yt-dlp)
  2. Extract audio as mono 16kHz WAV (ffmpeg)
  3. Transcribe audio (Whisper) → transcript/
  4. Separate speech vs car noise (SpeechBrain or fallback) → stems/
"""

import subprocess
from pathlib import Path
import shutil
import sys
import os

# -----------------------
# Environment check
# -----------------------
REQUIRED_ENV = "rally-audio"
current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
if current_env != REQUIRED_ENV:
    print(f"""
[ERROR] Wrong conda environment
  Current : {current_env or '(none)'}
  Required: {REQUIRED_ENV}

Fix:
  conda activate {REQUIRED_ENV}
  python download.py
""")
    sys.exit(1)

# -----------------------
# Executable check
# -----------------------
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
        print("\nActivate the correct conda environment and install missing tools.")
        sys.exit(1)

check_executables()

# -----------------------
# Configuration
# -----------------------
VIDEO_URLS = [
    "https://www.youtube.com/watch?v=5gJxWIybplc",
    # Add more YouTube links here
]

WHISPER_MODEL  = "small"   # tiny / base / small / medium / large
WHISPER_DEVICE = "cpu"     # cpu or cuda
ENABLE_WHISPER = True      # Set False to skip transcription

# -----------------------
# Helper
# -----------------------
def run(cmd: list, label: str = ""):
    tag = f"[{label}] " if label else ""
    print(f"\n{tag}Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n{tag}[ERROR] Command failed with exit code {result.returncode}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def conda_python() -> str:
    """
    Return the Python interpreter inside the active conda environment.
    sys.executable can resolve to the wrong Python (e.g. /opt/homebrew/python3.14)
    when conda's shims are not on PATH. CONDA_PREFIX always points to the active env.
    """
    conda_root = os.environ.get("CONDA_PREFIX", "")
    if conda_root:
        candidate = Path(conda_root) / "bin" / "python"
        if candidate.exists():
            return str(candidate)

    # Fallback: ask conda which python is in the named env
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

    print("[WARN] Could not resolve conda Python via CONDA_PREFIX; falling back to sys.executable")
    return sys.executable

# -----------------------
# Directory setup
# -----------------------
for d in ["raw", "transcript", "stems"]:
    Path(d).mkdir(exist_ok=True)

# -----------------------
# Main loop
# -----------------------
for idx, url in enumerate(VIDEO_URLS, start=1):
    print(f"\n{'='*60}")
    print(f"  Processing video {idx}/{len(VIDEO_URLS)}")
    print(f"  URL: {url}")
    print(f"{'='*60}")

    video_file = f"raw/video_{idx}.mp4"
    audio_file = f"raw/audio_{idx}.wav"

    # --------------------------------------------------
    # Step 1: Download video
    # --------------------------------------------------
    print("\n[Step 1/4] Downloading video...")
    run([
        "yt-dlp",
        "-f", "bestvideo+bestaudio",
        "-o", video_file.replace(".mp4", ".%(ext)s"),
        "--merge-output-format", "mp4",
        "--no-playlist",
        url
    ], "yt-dlp")

    # yt-dlp may produce video_{idx}.mp4 or video_{idx}.webm etc — resolve actual file
    actual_video = None
    for ext in ["mp4", "mkv", "webm"]:
        candidate = f"raw/video_{idx}.{ext}"
        if Path(candidate).exists():
            actual_video = candidate
            break
    if not actual_video:
        print(f"[ERROR] Could not find downloaded video file for index {idx}")
        continue

    # --------------------------------------------------
    # Step 2: Extract audio
    # --------------------------------------------------
    # Extract at 48kHz — matches DeepFilterNet's native sample rate so it
    # receives real full-bandwidth signal instead of zero-padded upsamples.
    # loudnorm: EBU R128 loudness normalization for consistent input levels.
    # Whisper will receive a 16kHz downsample of the enhanced speech stem
    # (produced by speechbrain_split.py) rather than this raw 48kHz file.
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

    # --------------------------------------------------
    # Step 3: Separate speech vs car noise
    # --------------------------------------------------
    print("\n[Step 3/4] Separating speech and car noise...")
    stems_dir = f"stems/video_{idx}"
    Path(stems_dir).mkdir(parents=True, exist_ok=True)

    split_script = Path(__file__).parent / "speechbrain_split.py"
    if not split_script.exists():
        print(f"[ERROR] speechbrain_split.py not found at {split_script}")
        sys.exit(1)

    run([
        conda_python(),
        str(split_script),
        audio_file,
        stems_dir,
    ], "separation")

    # --------------------------------------------------
    # Step 4: Transcribe with Whisper
    # --------------------------------------------------
    # Whisper needs 16kHz. We transcribe the enhanced speech stem (not the
    # raw audio) — this gives Whisper a cleaner, noise-suppressed signal.
    speech_stem    = f"{stems_dir}/speech.wav"
    speech_16k     = f"{stems_dir}/speech_16k.wav"

    if ENABLE_WHISPER:
        print("\n[Step 4/4] Downsampling speech stem and transcribing with Whisper...")

        # Downsample enhanced speech to 16kHz for Whisper
        run([
            "ffmpeg", "-y",
            "-i", speech_stem,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            speech_16k
        ], "ffmpeg")

        transcript_dir = f"transcript/video_{idx}"
        Path(transcript_dir).mkdir(parents=True, exist_ok=True)
        run([
            "whisper", speech_16k,
            "--model",            WHISPER_MODEL,
            "--word_timestamps",  "True",
            "--output_format",    "all",
            "--output_dir",       transcript_dir,
            "--device",           WHISPER_DEVICE,
        ], "whisper")
        print(f"  Transcripts saved to: {transcript_dir}/")
    else:
        print("\n[Step 4/4] Whisper transcription skipped (ENABLE_WHISPER=False)")

    print(f"\n  Done! Outputs for video {idx}:")
    print(f"    Video      : {actual_video}")
    print(f"    Raw audio  : {audio_file}  (48kHz, normalized)")
    print(f"    Speech     : {stems_dir}/speech.wav  (48kHz, DeepFilterNet enhanced)")
    print(f"    Noise      : {stems_dir}/noise.wav   (48kHz, speech suppressed)")
    if ENABLE_WHISPER:
        print(f"    Speech 16k : {speech_16k}  (for Whisper)")
        print(f"    Transcript : transcript/video_{idx}/")

print(f"\n{'='*60}")
print("  All videos processed successfully!")
print(f"{'='*60}\n")