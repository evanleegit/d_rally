# config.py — Rally Audio Pipeline Configuration
# ------------------------------------------------
# Each team member edits this file once before running anything.
# Do not commit your personal config to a shared repo.

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Box folder path
# ─────────────────────────────────────────────
# Set this to your local Box sync folder for the group.
# Outputs (stems, transcripts, analysis figures) will be copied here after
# each clip is processed.
#
# Mac example   : /Users/yourname/Library/CloudStorage/Box-Box/GroupD
# Windows example: C:/Users/yourname/Box/GroupD
# Linux example  : /home/yourname/Box/GroupD
#
# If you leave this as None, outputs stay local only (no Box export).

BOX_FOLDER = "/Users/evanlee/Library/CloudStorage/Box-Box/GroupD"

# ─────────────────────────────────────────────
# Whisper settings
# ─────────────────────────────────────────────
WHISPER_MODEL  = "small"   # tiny / base / small / medium / large
WHISPER_DEVICE = "cpu"     # cpu | cuda (NVIDIA GPU) | mps (Apple Silicon)
ENABLE_WHISPER = True      # set False to skip transcription entirely

# ─────────────────────────────────────────────
# Conda environment name
# ─────────────────────────────────────────────
CONDA_ENV = "rally-audio"

# ─────────────────────────────────────────────
# Validation (runs on import — do not edit below this line)
# ─────────────────────────────────────────────
def get_box_path() -> Path | None:
    """
    Returns the resolved Box output path, or None if not configured / not found.
    Prints a warning if BOX_FOLDER is set but doesn't exist on this machine.
    """
    if BOX_FOLDER is None:
        return None
    p = Path(BOX_FOLDER).expanduser()
    if not p.exists():
        print(f"[config] WARNING: Box folder not found at {p}")
        print("         Outputs will be saved locally only.")
        print("         Update BOX_FOLDER in config.py to match your machine.")
        return None
    return p