import librosa
import numpy as np
import os
import noisereduce as nr

def load_audio(file_path, sr=16000, duration=5):
    """
    Loads an audio file and pads/truncates it to a fixed duration.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        # Pad if shorter than duration
        target_length = int(sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def reduce_noise(audio, sr=16000):
    """
    Applies moderate noise reduction to audio signal.
    Uses stationary noise reduction to preserve speech characteristics.
    """
    try:
        # Apply noise reduction with moderate settings
        reduced_noise = nr.reduce_noise(
            y=audio, 
            sr=sr,
            stationary=True,  # Assumes stationary background noise
            prop_decrease=0.8  # Moderate reduction (0.0 = no reduction, 1.0 = max reduction)
        )
        return reduced_noise
    except Exception as e:
        print(f"Warning: Noise reduction failed: {e}. Using original audio.")
        return audio

def extract_mel_spectrogram(file_path, sr=16000, n_mels=128, duration=5, apply_noise_reduction=True):
    """
    Extracts Mel Spectrogram from an audio file.
    Returns a numpy array of shape (n_mels, time_steps).
    
    Args:
        file_path: Path to audio file
        sr: Sample rate
        n_mels: Number of mel bands
        duration: Duration in seconds
        apply_noise_reduction: Whether to apply noise reduction (default: True)
    """
    y = load_audio(file_path, sr=sr, duration=duration)
    if y is None:
        return None
    
    # Apply noise reduction if enabled
    if apply_noise_reduction:
        y = reduce_noise(y, sr=sr)
    
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_db

def create_dummy_dataset(num_samples=20, save_dir='data/dummy'):
    """
    Creates dummy .wav files for testing the pipeline.
    """
    import soundfile as sf
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    labels = []
    files = []
    
    for i in range(num_samples):
        # Generate random noise as "audio"
        sr = 16000
        duration = 5
        y = np.random.uniform(-1, 1, int(sr * duration))
        
        # Label 0 for real, 1 for fake (arbitrary assignment for dummy data)
        label = i % 2
        filename = os.path.join(save_dir, f"sample_{i}_{label}.wav")
        sf.write(filename, y, sr)
        
        files.append(filename)
        labels.append(label)
        
    return files, labels
