"""
speechbrain_split.py - Separate co-driver speech from car/road noise
---------------------------------------------------------------------
Usage:
    python speechbrain_split.py <input_wav> <output_dir>

Strategy — two fully independent stems:

  speech.wav  ← DeepFilterNet enhanced output, untouched.
                Best possible voice signal — what the model is trained for.

  noise.wav   ← Speech-informed magnitude suppression on the original STFT.
                Uses the DeepFilterNet speech estimate to compute a per-bin
                speech fraction mask, then aggressively suppresses those bins
                in the original, reconstructing with the original phase.
                Produces a continuous, natural-sounding engine track with
                no gating, no phase cancellation, no freezing.

Install: pip install deepfilternet soundfile
"""

import sys
import os
import numpy as np
from pathlib import Path
import scipy.signal as signal
import scipy.io.wavfile as wavfile


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def load_mono_float(path: str):
    sr, data = wavfile.read(path)
    data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    peak = np.max(np.abs(data))
    if peak > 0:
        data /= peak
    return sr, data

def save_wav(path: Path, data: np.ndarray, sr: int):
    wavfile.write(str(path), sr,
                  np.clip(data * 32767, -32768, 32767).astype(np.int16))


# ─────────────────────────────────────────────
# Noise stem: speech-informed magnitude suppression
# ─────────────────────────────────────────────

def extract_noise_stem(original: np.ndarray, speech_enhanced: np.ndarray,
                       sr: int, out_path: Path):
    """
    Build a speech-free noise track using speech-informed STFT suppression.

    For each time-frequency bin (t, f):
      speech_fraction = |STFT(speech)| / (|STFT(original)| + eps)

    This tells us exactly how much of each bin's energy is speech vs engine.
    We then suppress bins proportional to their speech fraction, aggressively
    in the voice band, using a raised oversubtraction factor (beta).

    Crucially: reconstruction uses the ORIGINAL signal's phase throughout.
    No phase misalignment, no cancellation artifacts, continuous output.

    Tuning knobs:
      beta        — overall suppression strength (1.0=gentle, 3.0=aggressive)
      voice_boost — extra suppression multiplier in the 300–3400Hz voice band
      floor       — minimum gain (0=silent zeros, 0.05=keeps faint engine hum)
    """
    # ── STFT parameters ──────────────────────────────────────────────────
    # Larger nperseg = better frequency resolution for voice vs engine separation
    nperseg  = 1024
    noverlap = 896   # 87.5% overlap for smooth reconstruction

    # Match lengths
    min_len       = min(len(original), len(speech_enhanced))
    original      = original[:min_len]
    speech_enhanced = speech_enhanced[:min_len]

    # STFT of both signals
    freqs, _, Z_orig  = signal.stft(original,        fs=sr, nperseg=nperseg,
                                     noverlap=noverlap, window='hann')
    _,     _, Z_speech = signal.stft(speech_enhanced, fs=sr, nperseg=nperseg,
                                      noverlap=noverlap, window='hann')

    mag_orig   = np.abs(Z_orig)    # (freq_bins, time_frames)
    mag_speech = np.abs(Z_speech)
    phase_orig = np.angle(Z_orig)  # always use original phase

    eps = 1e-8

    # ── Speech fraction per bin ───────────────────────────────────────────
    # How much of each bin's magnitude is speech?
    speech_fraction = mag_speech / (mag_orig + eps)
    speech_fraction = np.clip(speech_fraction, 0.0, 1.0)

    # ── Suppression parameters ────────────────────────────────────────────
    beta        = 2.5   # overall oversubtraction — raise if speech still audible
    voice_boost = 1.8   # extra multiplier in voice band (300–3400Hz)
    floor       = 0.05  # minimum gain — prevents total zeroing / musical noise

    # Build frequency-dependent beta (boost suppression in voice band)
    beta_freq       = np.ones(len(freqs)) * beta
    voice_band_mask = (freqs >= 300) & (freqs <= 3400)
    beta_freq[voice_band_mask] *= voice_boost
    beta_freq       = beta_freq[:, np.newaxis]  # (freq_bins, 1) for broadcasting

    # ── Wiener-style gain for noise stem ─────────────────────────────────
    # gain = max(floor, 1 - beta * speech_fraction)
    # Where speech fraction is high → gain near floor (suppress)
    # Where speech fraction is low  → gain near 1.0  (keep engine)
    gain = np.maximum(floor, 1.0 - beta_freq * speech_fraction)

    # Apply gain to original magnitude, reconstruct with original phase
    mag_noise = mag_orig * gain
    Z_noise   = mag_noise * np.exp(1j * phase_orig)

    _, noise_td = signal.istft(Z_noise, fs=sr, nperseg=nperseg,
                                noverlap=noverlap, window='hann')
    noise_td = noise_td[:len(original)]

    # Normalize output
    peak = np.max(np.abs(noise_td))
    if peak > 0:
        noise_td /= peak

    save_wav(out_path / "noise.wav", noise_td, sr)

    # Diagnostics
    avg_gain_voice  = np.mean(gain[voice_band_mask, :])
    avg_gain_engine = np.mean(gain[~voice_band_mask, :])
    print(f"[Noise]  Average gain in voice band:  {avg_gain_voice:.3f}  "
          f"(lower = more suppression)")
    print(f"[Noise]  Average gain outside voice:  {avg_gain_engine:.3f}")
    print(f"[Noise]  noise → {out_path / 'noise.wav'}")


# ─────────────────────────────────────────────
# Method 1: DeepFilterNet (primary)
# ─────────────────────────────────────────────

def separate_with_deepfilter(input_wav: str, output_dir: str) -> bool:
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
    except ImportError:
        print("[DeepFilterNet] Not installed.  pip install deepfilternet")
        return False

    try:
        import soundfile as sf
        from scipy.signal import resample_poly
        from math import gcd

        print("[DeepFilterNet] Loading model...")
        model, df_state, _ = init_df()
        model.eval()

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Load original at its native sample rate
        orig_np, orig_sr = sf.read(input_wav)
        orig_np = orig_np.astype(np.float32)
        if orig_np.ndim > 1:
            orig_np = orig_np.mean(axis=1)
        peak = np.max(np.abs(orig_np))
        if peak > 0:
            orig_np /= peak

        # Run DeepFilterNet at its native rate (48kHz)
        print("[Speech]  Running DeepFilterNet...")
        audio_df = load_audio(input_wav, sr=df_state.sr())
        if isinstance(audio_df, tuple):
            audio_df, _ = audio_df
        enhanced = enhance(model, df_state, audio_df)

        # Convert to numpy
        enh_np = enhanced.squeeze()
        enh_np = enh_np.numpy() if hasattr(enh_np, 'numpy') else np.array(enh_np)

        df_sr = df_state.sr()  # 48000

        # Resample enhanced speech to original sample rate if needed
        if df_sr != orig_sr:
            g      = gcd(orig_sr, df_sr)
            enh_rs = resample_poly(enh_np, orig_sr // g, df_sr // g)
        else:
            enh_rs = enh_np.copy()

        # Trim/pad to exactly match original length
        if len(enh_rs) > len(orig_np):
            enh_rs = enh_rs[:len(orig_np)]
        elif len(enh_rs) < len(orig_np):
            enh_rs = np.pad(enh_rs, (0, len(orig_np) - len(enh_rs)))

        # Normalize and save speech stem
        sp_peak = np.max(np.abs(enh_rs))
        if sp_peak > 0:
            enh_rs /= sp_peak
        save_wav(out_path / "speech.wav", enh_rs, orig_sr)
        print(f"[Speech]  speech → {out_path / 'speech.wav'}  ({orig_sr}Hz)")

        # Build noise stem using speech as suppression reference
        print("[Noise]   Building speech-suppressed noise track...")
        extract_noise_stem(orig_np, enh_rs, orig_sr, out_path)
        print(f"          ({orig_sr}Hz — continuous, no gating)")
        return True

    except Exception as e:
        print(f"[DeepFilterNet] Error: {e}")
        import traceback; traceback.print_exc()
        return False


# ─────────────────────────────────────────────
# Method 2: HPSS + bandpass fallback
# ─────────────────────────────────────────────

def separate_spectral_fallback(input_wav: str, output_dir: str) -> None:
    try:
        import librosa
        import librosa.effects
        _use_hpss = True
        print("[Fallback] Using HPSS...")
    except ImportError:
        _use_hpss = False
        print("[Fallback] Using bandpass filter...")

    sr, audio = load_mono_float(input_wav)
    out_path  = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if _use_hpss:
        harmonic, percussive = librosa.effects.hpss(audio, margin=3.0)
        nyq  = sr / 2.0
        b, a = signal.butter(4, [200/nyq, 3500/nyq], btype='band')
        voice_band = signal.filtfilt(b, a, harmonic)
        speech = percussive + voice_band * 0.5
        noise  = harmonic   - voice_band * 0.5
    else:
        nyq    = sr / 2.0
        b, a   = signal.butter(5, [max(150/nyq, 0.001), min(4000/nyq, 0.999)], btype='band')
        speech = signal.filtfilt(b, a, audio)
        noise  = audio - speech

    save_wav(out_path / "speech.wav", speech, sr)
    save_wav(out_path / "noise.wav",  noise,  sr)
    print(f"[Fallback] speech → {out_path / 'speech.wav'}")
    print(f"[Fallback] noise  → {out_path / 'noise.wav'}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("Usage: python speechbrain_split.py <input_wav> <output_dir>")
        sys.exit(1)

    input_wav  = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isfile(input_wav):
        print(f"[ERROR] Input file not found: {input_wav}")
        sys.exit(1)

    print(f"\nInput  : {input_wav}")
    print(f"Output : {output_dir}\n")

    success = separate_with_deepfilter(input_wav, output_dir)
    if not success:
        print("\n[INFO] DeepFilterNet unavailable — using spectral fallback...")
        separate_spectral_fallback(input_wav, output_dir)

    print(f"\n✓ Separation complete → {output_dir}/")
    print("    speech.wav  — co-driver voice (DeepFilterNet enhanced)")
    print("    noise.wav   — engine/road noise (continuous, speech suppressed)")


if __name__ == "__main__":
    main()