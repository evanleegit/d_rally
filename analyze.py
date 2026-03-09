"""
analyze.py - Rally Audio Analysis & Visualization
--------------------------------------------------
Usage:
    python analyze.py <original_wav> <speech_wav> <noise_wav> [output_png]

Example:
    python analyze.py raw/audio_1.wav stems/video_1/speech.wav stems/video_1/noise.wav

Or run automatically for all processed videos:
    python analyze.py --all

Produces a 4-panel figure per video:
  1. Spectrogram — original audio (pre-split)
  2. Spectrogram — speech stem
  3. Spectrogram — noise stem
  4. Time series  — speech RMS amplitude vs noise spectral centroid (RPM proxy)

Output saved to: analysis/video_N.png
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for all environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator


# ─────────────────────────────────────────────
# Audio loading
# ─────────────────────────────────────────────

def load(path: str):
    """Load WAV as mono float32, normalized to [-1, 1]."""
    sr, data = wavfile.read(path)
    data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    peak = np.max(np.abs(data))
    if peak > 0:
        data /= peak
    return sr, data


# ─────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────

def compute_spectrogram(audio: np.ndarray, sr: int,
                         frame_ms: int = 50,
                         overlap_pct: float = 0.75):
    """
    Compute STFT spectrogram.
    Returns (freqs, times, power_db) where power_db is in dB.
    frame_ms=50 gives ~20 frames/sec — fine enough for rally pace note timing.
    """
    nperseg  = int(sr * frame_ms / 1000)
    noverlap = int(nperseg * overlap_pct)
    freqs, times, Z = signal.stft(audio, fs=sr, nperseg=nperseg,
                                   noverlap=noverlap, window='hann')
    power = np.abs(Z) ** 2
    power_db = 10 * np.log10(power + 1e-10)
    return freqs, times, power_db


def compute_rms_envelope(audio: np.ndarray, sr: int,
                          window_ms: int = 250,
                          hop_ms: int = 50):
    """
    Rolling RMS of the speech stem.
    window_ms=250 smooths individual words into call-level events.
    hop_ms=50 matches the spectrogram frame rate.
    Returns (times, rms).
    """
    window = int(sr * window_ms / 1000)
    hop    = int(sr * hop_ms    / 1000)
    times, rms = [], []
    for start in range(0, len(audio) - window, hop):
        chunk = audio[start:start + window]
        times.append((start + window / 2) / sr)
        rms.append(np.sqrt(np.mean(chunk ** 2)))
    return np.array(times), np.array(rms)


def compute_spectral_centroid(audio: np.ndarray, sr: int,
                               window_ms: int = 250,
                               hop_ms: int = 50,
                               f_low: float = 80.0,
                               f_high: float = 4000.0):
    """
    Rolling spectral centroid of the noise stem, limited to f_low–f_high Hz.
    Acts as a proxy for engine RPM:
      - Higher centroid → engine harmonics are higher → higher RPM
      - Lower centroid  → low rumble dominant → lower RPM / deceleration

    f_low=80Hz removes DC/sub-bass rumble.
    f_high=4000Hz excludes voice-band residual and focuses on engine harmonics.
    Returns (times, centroid_hz).
    """
    window = int(sr * window_ms / 1000)
    hop    = int(sr * hop_ms    / 1000)
    times, centroids = [], []

    for start in range(0, len(audio) - window, hop):
        chunk = audio[start:start + window]
        fft   = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1 / sr)

        # Restrict to engine band
        mask  = (freqs >= f_low) & (freqs <= f_high)
        f_band = freqs[mask]
        a_band = fft[mask]

        total = np.sum(a_band)
        centroid = np.sum(f_band * a_band) / (total + 1e-10)

        times.append((start + window / 2) / sr)
        centroids.append(centroid)

    return np.array(times), np.array(centroids)


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

# Palette — dark scientific instrument aesthetic
BG       = "#0d0f14"
PANEL_BG = "#13161e"
TEXT     = "#e8eaf0"
MUTED    = "#6b7280"
GRID     = "#1e2230"
ACCENT_S = "#4fc3f7"   # speech — cool blue
ACCENT_N = "#ff7043"   # noise centroid — warm orange
ACCENT_R = "#aed581"   # RMS fill — soft green


def plot_spectrogram(ax, freqs, times, power_db,
                     title: str, cmap: str, duration: float):
    """Render a single spectrogram panel."""
    # Clip dB range for visual clarity
    vmin = np.percentile(power_db, 5)
    vmax = np.percentile(power_db, 99)

    ax.pcolormesh(times, freqs, power_db,
                  shading='gouraud', cmap=cmap,
                  vmin=vmin, vmax=vmax, rasterized=True)

    ax.set_ylabel("Frequency (Hz)", color=TEXT, fontsize=8)
    ax.set_ylim(0, min(np.max(freqs), 8000))
    ax.set_xlim(0, duration)
    ax.set_facecolor(PANEL_BG)

    # Y-axis: label key frequency landmarks
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_minor_locator(MultipleLocator(500))
    ax.tick_params(axis='both', colors=MUTED, labelsize=7)

    # Panel label
    ax.text(0.01, 0.95, title, transform=ax.transAxes,
            color=TEXT, fontsize=9, fontweight='bold',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG,
                      edgecolor='none', alpha=0.7))

    # Horizontal guide lines at voice band edges
    for f, label in [(300, "300Hz"), (3400, "3.4kHz")]:
        if f <= 8000:
            ax.axhline(f, color=MUTED, linewidth=0.5, linestyle='--', alpha=0.5)
            ax.text(duration * 0.99, f + 80, label,
                    color=MUTED, fontsize=6, ha='right')


def make_figure(original_path: str, speech_path: str,
                noise_path: str, out_path: Path,
                video_label: str = ""):
    """
    Build and save the 4-panel analysis figure.
    """
    print(f"[analyze] Loading audio files...")
    sr_o, orig   = load(original_path)
    sr_s, speech = load(speech_path)
    sr_n, noise  = load(noise_path)

    duration = len(orig) / sr_o

    print(f"[analyze] Computing spectrograms...")
    f_o, t_o, S_o = compute_spectrogram(orig,   sr_o)
    f_s, t_s, S_s = compute_spectrogram(speech, sr_s)
    f_n, t_n, S_n = compute_spectrogram(noise,  sr_n)

    print(f"[analyze] Computing RMS envelope and spectral centroid...")
    rms_t,  rms_v  = compute_rms_envelope(speech, sr_s)
    cent_t, cent_v = compute_spectral_centroid(noise, sr_n)

    # Normalize both series to [0,1] for dual-axis overlay
    rms_norm  = rms_v  / (np.max(rms_v)  + 1e-10)
    cent_norm = cent_v / (np.max(cent_v) + 1e-10)

    # ── Layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        4, 1,
        figure=fig,
        hspace=0.08,
        top=0.93, bottom=0.07,
        left=0.07, right=0.97,
        height_ratios=[2, 2, 2, 1.6]
    )

    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    # ── Panel 1: Original spectrogram ─────────────────────────────────────
    plot_spectrogram(axes[0], f_o, t_o, S_o,
                     "Original Audio (pre-split)", "inferno", duration)

    # ── Panel 2: Speech stem spectrogram ──────────────────────────────────
    plot_spectrogram(axes[1], f_s, t_s, S_s,
                     "Speech Stem (co-driver voice)", "Blues_r", duration)

    # ── Panel 3: Noise stem spectrogram ───────────────────────────────────
    plot_spectrogram(axes[2], f_n, t_n, S_n,
                     "Noise Stem (engine / road)", "YlOrRd", duration)

    # ── Panel 4: Time series overlay ──────────────────────────────────────
    ax4 = axes[3]
    ax4_r = ax4.twinx()   # second y-axis for spectral centroid

    # Speech RMS — filled area
    ax4.fill_between(rms_t, rms_norm, alpha=0.35, color=ACCENT_R, linewidth=0)
    ax4.plot(rms_t, rms_norm, color=ACCENT_R, linewidth=1.2,
             label="Speech RMS (voice activity)")

    # Noise spectral centroid — line
    ax4_r.plot(cent_t, cent_norm, color=ACCENT_N, linewidth=1.5,
               label="Noise centroid (RPM proxy)", alpha=0.9)

    # Style both axes
    for ax in (ax4, ax4_r):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlim(0, duration)

    ax4.set_ylabel("Speech RMS\n(normalized)", color=ACCENT_R,
                   fontsize=8, labelpad=4)
    ax4_r.set_ylabel("Engine centroid\n(normalized)", color=ACCENT_N,
                     fontsize=8, labelpad=4)
    ax4.tick_params(axis='y', colors=ACCENT_R)
    ax4_r.tick_params(axis='y', colors=ACCENT_N)

    ax4.set_xlabel("Time (seconds)", color=TEXT, fontsize=9)
    ax4.set_facecolor(PANEL_BG)

    ax4.text(0.01, 0.93, "Voice Activity vs Engine Load",
             transform=ax4.transAxes, color=TEXT, fontsize=9,
             fontweight='bold', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=BG,
                       edgecolor='none', alpha=0.7))

    # Combined legend
    lines_l, labels_l = ax4.get_legend_handles_labels()
    lines_r, labels_r = ax4_r.get_legend_handles_labels()
    ax4.legend(lines_l + lines_r, labels_l + labels_r,
               loc='upper right', fontsize=7,
               facecolor=BG, edgecolor=GRID,
               labelcolor=TEXT, framealpha=0.85)

    # Grid on time-series panel only
    ax4.yaxis.grid(True, color=GRID, linewidth=0.5, linestyle='-')
    ax4.set_axisbelow(True)

    # X-axis ticks — hide on top 3 panels, show on bottom
    for ax in axes[:3]:
        ax.set_xticklabels([])
        ax.set_xlabel("")
        ax.tick_params(axis='x', colors=MUTED)

    # ── Title ─────────────────────────────────────────────────────────────
    title = f"Rally Audio Analysis  —  {video_label}" if video_label else \
            "Rally Audio Analysis"
    fig.text(0.5, 0.97, title, ha='center', va='top',
             color=TEXT, fontsize=13, fontweight='bold',
             fontfamily='monospace')

    subtitle = (f"Original: {Path(original_path).name}  |  "
                f"Duration: {duration:.1f}s  |  "
                f"Sample rate: {sr_o}Hz")
    fig.text(0.5, 0.955, subtitle, ha='center', va='top',
             color=MUTED, fontsize=7.5, fontfamily='monospace')

    # ── Save ──────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"[analyze] Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def run_all():
    """Auto-discover processed videos and analyze all of them."""
    stems_root = Path("stems")
    raw_root   = Path("raw")

    if not stems_root.exists():
        print("[ERROR] No stems/ directory found. Run download.py first.")
        sys.exit(1)

    video_dirs = sorted(stems_root.glob("video_*"))
    if not video_dirs:
        print("[ERROR] No video_* folders found in stems/. Run download.py first.")
        sys.exit(1)

    print(f"[analyze] Found {len(video_dirs)} video(s) to analyze.")
    for vdir in video_dirs:
        idx = vdir.name.split("_")[1]

        # Resolve paths
        speech_path = vdir / "speech.wav"
        noise_path  = vdir / "noise.wav"

        # Find the raw audio — could be 48kHz or 16kHz
        audio_path = raw_root / f"audio_{idx}.wav"

        if not speech_path.exists():
            print(f"[SKIP] {vdir.name}: speech.wav not found")
            continue
        if not noise_path.exists():
            print(f"[SKIP] {vdir.name}: noise.wav not found")
            continue
        if not audio_path.exists():
            print(f"[SKIP] {vdir.name}: raw audio not found at {audio_path}")
            continue

        out_path = Path("analysis") / f"video_{idx}.png"
        print(f"\n{'='*55}")
        print(f"  Analyzing video {idx}: {audio_path.name}")
        print(f"{'='*55}")
        make_figure(str(audio_path), str(speech_path), str(noise_path),
                    out_path, video_label=f"Video {idx}")


def main():
    # all flag: process everything automatically
    if len(sys.argv) == 2 and sys.argv[1] == "--all":
        run_all()
        return

    if len(sys.argv) < 4:
        print(__doc__)
        print("Usage: python analyze.py <original> <speech> <noise> [output.png]")
        print("       python analyze.py --all")
        sys.exit(1)

    original_path = sys.argv[1]
    speech_path   = sys.argv[2]
    noise_path    = sys.argv[3]
    out_path      = Path(sys.argv[4]) if len(sys.argv) > 4 \
                    else Path("analysis") / "output.png"

    for p in [original_path, speech_path, noise_path]:
        if not os.path.isfile(p):
            print(f"[ERROR] File not found: {p}")
            sys.exit(1)

    label = Path(original_path).stem
    make_figure(original_path, speech_path, noise_path, out_path, label)

    print(f"\n✓ Analysis complete → {out_path}")


if __name__ == "__main__":
    main()