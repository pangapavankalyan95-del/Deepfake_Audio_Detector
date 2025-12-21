
import os
import random
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

def generate_mixed_samples(num_samples=100, output_dir="data/mixed_samples"):
    """
    Generates mixed audio files by concatenating a Real and a Fake sample.
    Randomly chooses order: [Real, Fake] or [Fake, Real].
    """
    
    # Setup paths
    base_dir = Path("data/dataset/train")
    real_dir = base_dir / "real"
    fake_dir = base_dir / "fake"
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning for audio files in {base_dir}...")
    
    # Get list of files
    real_files = list(real_dir.glob("*.flac")) + list(real_dir.glob("*.wav"))
    fake_files = list(fake_dir.glob("*.flac")) + list(fake_dir.glob("*.wav"))
    
    if not real_files or not fake_files:
        print("Error: Could not find audio files in train/real or train/fake")
        return

    print(f"Found {len(real_files)} Real and {len(fake_files)} Fake files.")
    
    print(f"Generating {num_samples} mixed samples...")
    
    for i in tqdm(range(num_samples)):
        # 1. Select random pair
        r_file = random.choice(real_files)
        f_file = random.choice(fake_files)
        
        # 2. Load audio (resample to 16k standard)
        try:
            yg_r, sr = librosa.load(str(r_file), sr=16000)
            yg_f, _ = librosa.load(str(f_file), sr=16000)
            
            # 3. Randomize Order
            order = random.choice(["real_first", "fake_first"])
            
            if order == "real_first":
                # Real then Fake
                mixed_audio = np.concatenate([yg_r, yg_f])
                filename = f"mixed_{i+1}_real_fake.wav"
            else:
                # Fake then Real
                mixed_audio = np.concatenate([yg_f, yg_r])
                filename = f"mixed_{i+1}_fake_real.wav"
            
            # 4. Save
            save_path = output_path / filename
            sf.write(str(save_path), mixed_audio, sr)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    print(f"Successfully generated samples in {output_path}")

if __name__ == "__main__":
    generate_mixed_samples()
