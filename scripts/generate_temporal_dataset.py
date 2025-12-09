"""
Temporal Dataset Generation Script
Generates mixed audio samples (real + fake segments) for temporal deepfake detection validation.

HYBRID APPROACH:
- 30 samples: User's voice + Training fake
- 20 samples: Training real + Training fake

SAFETY:
- Only READS from existing data
- Only WRITES to data/temporal_validation/
- No modifications to original datasets
"""

import os
import json
import random
import shutil
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
TEMPORAL_DIR = Path("data/temporal_validation")
MIXED_AUDIO_DIR = TEMPORAL_DIR / "mixed_audio"
ANNOTATIONS_DIR = TEMPORAL_DIR / "annotations"
SOURCE_DIR = TEMPORAL_DIR / "source"
USER_RECORDINGS_DIR = SOURCE_DIR / "user_recordings"

TRAINING_REAL_DIR = Path("data/dataset/train/real")
TRAINING_FAKE_DIR = Path("data/dataset/train/fake")
SPEAKER_PROFILES_DIR = Path("data/speaker_profiles")

# Target counts
USER_VOICE_SAMPLES = 30
TRAINING_SAMPLES = 20
TOTAL_SAMPLES = USER_VOICE_SAMPLES + TRAINING_SAMPLES

# Audio parameters
TARGET_SR = 16000  # Sample rate
MIN_DURATION = 10  # Minimum audio duration in seconds (REDUCED for user recordings)
MAX_DURATION = 45  # Maximum audio duration in seconds
MIN_FAKE_SEGMENT = 3   # Minimum fake segment duration (REDUCED)
MAX_FAKE_SEGMENT = 15  # Maximum fake segment duration


def log_message(message):
    """Log messages with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def collect_user_recordings():
    """
    Collect user's recorded audio samples from speaker profiles.
    
    SAFETY: Only COPIES files, never modifies originals.
    """
    log_message("Collecting user recordings...")
    
    user_files = []
    
    # Check if speaker profiles directory exists
    if not SPEAKER_PROFILES_DIR.exists():
        log_message(f"Warning: {SPEAKER_PROFILES_DIR} not found. Skipping user recordings.")
        return user_files
    
    # Look for .pkl files (speaker profiles)
    for pkl_file in SPEAKER_PROFILES_DIR.glob("*.pkl"):
        log_message(f"Found speaker profile: {pkl_file.name}")
    
    # Look for any .wav files in the directory
    for wav_file in SPEAKER_PROFILES_DIR.glob("*.wav"):
        user_files.append(wav_file)
        log_message(f"Found user recording: {wav_file.name}")
    
    # Also check parent directory for any debug recordings
    project_root = Path(".")
    for wav_file in project_root.glob("temp_*.wav"):
        if wav_file.exists():
            user_files.append(wav_file)
            log_message(f"Found temp recording: {wav_file.name}")
    
    log_message(f"Total user recordings found: {len(user_files)}")
    return user_files


def load_audio(file_path, target_sr=TARGET_SR):
    """Load audio file and resample if needed."""
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        return audio, sr
    except Exception as e:
        log_message(f"Error loading {file_path}: {e}")
        return None, None


def mix_audio_segments(real_audio, fake_audio, sr=TARGET_SR):
    """
    Mix real and fake audio segments.
    
    Args:
        real_audio: Real audio array
        fake_audio: Fake audio array
        sr: Sample rate
        
    Returns:
        mixed_audio, segments_info
    """
    real_duration = len(real_audio) / sr
    fake_duration = len(fake_audio) / sr
    
    # If real audio is too short, repeat it to reach minimum duration
    if real_duration < MIN_DURATION:
        repeats_needed = int(np.ceil(MIN_DURATION / real_duration))
        real_audio = np.tile(real_audio, repeats_needed)
        real_duration = len(real_audio) / sr
        log_message(f"Concatenated audio {repeats_needed} times to reach {real_duration:.1f}s")
    
    # Trim or pad real audio to target duration
    target_duration = min(real_duration, random.uniform(MIN_DURATION, MAX_DURATION))
    target_samples = int(target_duration * sr)
    real_audio = real_audio[:target_samples]
    
    # Determine fake segment duration
    fake_segment_duration = min(fake_duration, random.uniform(MIN_FAKE_SEGMENT, MAX_FAKE_SEGMENT))
    fake_segment_samples = int(fake_segment_duration * sr)
    fake_segment = fake_audio[:fake_segment_samples]
    
    # Randomly choose insertion point (not at the very start or end)
    margin = int(2 * sr)  # 2 second margin (REDUCED from 5)
    if target_samples < (fake_segment_samples + 2 * margin):
        log_message("Audio too short for safe insertion, skipping...")
        return None, None
    
    insert_start_sample = random.randint(margin, target_samples - fake_segment_samples - margin)
    insert_end_sample = insert_start_sample + fake_segment_samples
    
    # Create mixed audio
    mixed_audio = real_audio.copy()
    mixed_audio[insert_start_sample:insert_end_sample] = fake_segment
    
    # Calculate timestamps
    insert_start_time = insert_start_sample / sr
    insert_end_time = insert_end_sample / sr
    total_duration = len(mixed_audio) / sr
    
    # Create segments info
    segments = [
        {"start": 0.0, "end": round(insert_start_time, 2), "label": "real"},
        {"start": round(insert_start_time, 2), "end": round(insert_end_time, 2), "label": "fake"},
        {"start": round(insert_end_time, 2), "end": round(total_duration, 2), "label": "real"}
    ]
    
    return mixed_audio, segments


def generate_user_voice_samples(user_files, fake_files, count=USER_VOICE_SAMPLES):
    """Generate samples with user's voice + training fake."""
    log_message(f"\nGenerating {count} user voice samples...")
    
    if len(user_files) == 0:
        log_message("No user recordings available. Skipping user voice samples.")
        return 0
    
    generated = 0
    attempts = 0
    max_attempts = count * 3  # Allow multiple attempts
    
    while generated < count and attempts < max_attempts:
        attempts += 1
        
        # Select random user recording and fake audio
        user_file = random.choice(user_files)
        fake_file = random.choice(fake_files)
        
        # Load audio
        real_audio, sr = load_audio(user_file)
        fake_audio, _ = load_audio(fake_file)
        
        if real_audio is None or fake_audio is None:
            continue
        
        # Mix audio
        mixed_audio, segments = mix_audio_segments(real_audio, fake_audio, sr)
        
        if mixed_audio is None:
            continue
        
        # Save mixed audio
        output_filename = f"user_voice_{generated+1:03d}.wav"
        output_path = MIXED_AUDIO_DIR / output_filename
        sf.write(output_path, mixed_audio, sr)
        
        # Save annotation
        annotation = {
            "filename": output_filename,
            "duration": round(len(mixed_audio) / sr, 2),
            "mix_type": "user_voice",
            "segments": segments,
            "source_files": {
                "real": user_file.name,
                "fake": fake_file.name
            },
            "generated_at": datetime.now().isoformat()
        }
        
        annotation_path = ANNOTATIONS_DIR / f"user_voice_{generated+1:03d}.json"
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        generated += 1
        log_message(f"Generated user_voice_{generated:03d}.wav")
    
    log_message(f"Successfully generated {generated}/{count} user voice samples")
    return generated


def generate_training_samples(real_files, fake_files, count=TRAINING_SAMPLES):
    """Generate samples with training real + training fake."""
    log_message(f"\nGenerating {count} training samples...")
    
    generated = 0
    attempts = 0
    max_attempts = count * 3
    
    while generated < count and attempts < max_attempts:
        attempts += 1
        
        # Select random real and fake audio
        real_file = random.choice(real_files)
        fake_file = random.choice(fake_files)
        
        # Load audio
        real_audio, sr = load_audio(real_file)
        fake_audio, _ = load_audio(fake_file)
        
        if real_audio is None or fake_audio is None:
            continue
        
        # Mix audio
        mixed_audio, segments = mix_audio_segments(real_audio, fake_audio, sr)
        
        if mixed_audio is None:
            continue
        
        # Save mixed audio
        output_filename = f"training_{generated+1:03d}.wav"
        output_path = MIXED_AUDIO_DIR / output_filename
        sf.write(output_path, mixed_audio, sr)
        
        # Save annotation
        annotation = {
            "filename": output_filename,
            "duration": round(len(mixed_audio) / sr, 2),
            "mix_type": "training",
            "segments": segments,
            "source_files": {
                "real": real_file.name,
                "fake": fake_file.name
            },
            "generated_at": datetime.now().isoformat()
        }
        
        annotation_path = ANNOTATIONS_DIR / f"training_{generated+1:03d}.json"
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        generated += 1
        log_message(f"Generated training_{generated:03d}.wav")
    
    log_message(f"Successfully generated {generated}/{count} training samples")
    return generated


def main():
    """Main execution function."""
    log_message("="*60)
    log_message("TEMPORAL DATASET GENERATION SCRIPT")
    log_message("="*60)
    
    # Safety check
    log_message("\n[SAFETY CHECK]")
    log_message(f"✓ Will only READ from: {TRAINING_REAL_DIR}, {TRAINING_FAKE_DIR}")
    log_message(f"✓ Will only WRITE to: {TEMPORAL_DIR}")
    log_message(f"✓ Original data will NOT be modified")
    
    # Verify directories exist
    if not TRAINING_REAL_DIR.exists():
        log_message(f"ERROR: {TRAINING_REAL_DIR} not found!")
        return
    
    if not TRAINING_FAKE_DIR.exists():
        log_message(f"ERROR: {TRAINING_FAKE_DIR} not found!")
        return
    
    # Collect source files
    log_message("\n[COLLECTING SOURCE FILES]")
    user_files = collect_user_recordings()
    real_files = list(TRAINING_REAL_DIR.glob("*.flac")) + list(TRAINING_REAL_DIR.glob("*.wav"))
    fake_files = list(TRAINING_FAKE_DIR.glob("*.flac")) + list(TRAINING_FAKE_DIR.glob("*.wav"))
    
    log_message(f"Training real files: {len(real_files)}")
    log_message(f"Training fake files: {len(fake_files)}")
    
    if len(real_files) == 0 or len(fake_files) == 0:
        log_message("ERROR: Insufficient training data!")
        return
    
    # Generate datasets
    log_message("\n[GENERATING MIXED SAMPLES]")
    user_count = generate_user_voice_samples(user_files, fake_files)
    training_count = generate_training_samples(real_files, fake_files)
    
    # Summary
    log_message("\n" + "="*60)
    log_message("GENERATION COMPLETE")
    log_message("="*60)
    log_message(f"User voice samples: {user_count}/{USER_VOICE_SAMPLES}")
    log_message(f"Training samples: {training_count}/{TRAINING_SAMPLES}")
    log_message(f"Total samples: {user_count + training_count}/{TOTAL_SAMPLES}")
    log_message(f"\nOutput directory: {MIXED_AUDIO_DIR}")
    log_message(f"Annotations directory: {ANNOTATIONS_DIR}")
    log_message("="*60)


if __name__ == "__main__":
    main()
