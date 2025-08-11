#!/usr/bin/env python3
# ==============================================================================
# F5-TTS STANDALONE WORKER (OPTIMIZED)
# ==============================================================================
# This script is an optimized, standalone command-line tool for generating
# audio from a text file using the F5-TTS model.
#
# Usage:
# python f5_tts_worker.py <input_text_file.txt> <output_audio_file.wav> [path_to_voice_sample.wav]
# ==============================================================================

import sys
import os
import traceback
import re

import torch
import torchaudio
import numpy as np
from pydub import AudioSegment

try:
    from f5_tts.api import F5TTS
except ImportError:
    print("Error: F5-TTS library not found. Please ensure it is installed correctly.")
    sys.exit(1)

def flush(msg):
    """Prints a message to the console immediately."""
    print(msg, flush=True)

def validate_and_save_audio(wav, sr, outwav_path):
    """Validates, converts to 16-bit PCM, saves, and verifies the audio file."""
    if not isinstance(wav, (torch.Tensor, np.ndarray)) or wav.size == 0:
        return False, "Model output is not a valid or is an empty audio array."

    wav_tensor = torch.from_numpy(wav) if isinstance(wav, np.ndarray) else wav

    if torch.isnan(wav_tensor).any() or torch.isinf(wav_tensor).any():
        return False, "Audio tensor contains NaN or Inf values."
    if wav_tensor.numel() < sr * 0.05:
        return False, "Audio output is too short (less than 0.05s)."
    if torch.max(torch.abs(wav_tensor)) < 1e-6:
        return False, "Audio output is essentially silent."

    if wav_tensor.ndim == 1:
        wav_tensor = wav_tensor.unsqueeze(0)

    if wav_tensor.dtype.is_floating_point:
        flush("Converting float audio to 16-bit PCM format.")
        wav_tensor = (wav_tensor * 32767).clamp(-32768, 32767).to(torch.int16)

    try:
        os.makedirs(os.path.dirname(outwav_path), exist_ok=True)
        torchaudio.save(outwav_path, wav_tensor.cpu(), sr, format="wav", backend="soundfile")
        if not os.path.exists(outwav_path) or os.path.getsize(outwav_path) < 1024:
            return False, "Saved WAV file is missing or too small after conversion."
    except Exception as e:
        return False, f"Error during torchaudio.save: {e}"

    try:
        audio = AudioSegment.from_wav(outwav_path)
        if audio.sample_width != 2:
            os.remove(outwav_path)
            return False, f"CRITICAL: Audio format error. Expected sample width 2, but got {audio.sample_width}."
    except Exception as e:
        return False, f"CRITICAL: Final validation of saved WAV file failed: {e}"

    return True, "Audio saved and validated successfully in 16-bit PCM format."

if __name__ == '__main__':
    if len(sys.argv) < 3:
        flush("Usage: python f5_tts_worker.py <textfile> <output_wav> [voice_sample_path]")
        sys.exit(1)

    textfile, outwav = sys.argv[1], sys.argv[2]
    voice_sample_path = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        with open(textfile, "r", encoding="utf-8") as f:
            text = f.read().strip()
        text = re.sub(r'\s+', ' ', text).replace('–', '-').replace('’', "'")
        if not text:
            flush("Error: Input text is empty after normalization."); sys.exit(1)
    except Exception as e:
        flush(f"Error reading text file: {e}"); traceback.print_exc(); sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    flush(f"Using device: {device}")

    try:
        model = F5TTS(device=device, ode_method='euler')
        if device == "cuda" and hasattr(model, 'ema_model'):
            flush("Applying optimizations: half-precision (FP16) and torch.compile()...")
            model.ema_model.half()
            model.ema_model = torch.compile(model.ema_model, mode="reduce-overhead")
            flush("Model optimized successfully.")
    except Exception:
        flush("ERROR: Failed to load or optimize the F5TTS model."); traceback.print_exc(); sys.exit(1)

    try:
        flush("Transcribing reference audio (if needed)...")
        ref_text = "" 
        
        flush("Generating speech...")
        wav, sr, _ = model.infer(ref_file=voice_sample_path, ref_text=ref_text, gen_text=text, remove_silence=True)
    except Exception:
        flush("ERROR DURING SYNTHESIS:"); traceback.print_exc(); sys.exit(1)

    try:
        success, message = validate_and_save_audio(wav, sr, outwav)
        if not success:
            flush(f"ERROR: {message}"); sys.exit(1)
        flush(f"✅ Success! Generated audio saved to: {outwav}")
    except Exception:
        flush("ERROR SAVING AUDIO:"); traceback.print_exc(); sys.exit(1)
    
    sys.exit(0)
