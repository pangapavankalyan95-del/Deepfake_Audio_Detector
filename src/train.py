import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

from src.preprocess import create_dummy_dataset, extract_mel_spectrogram
from src.model import DeepfakeDetector


class AudioDataGenerator(tf.keras.utils.Sequence):
    """
    Memory-efficient data generator for large audio datasets.
    Loads and processes audio files on-the-fly instead of loading all into RAM.
    """
    def __init__(self, file_paths, labels, batch_size=16, shuffle=True, target_shape=(128, 157), augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_shape = target_shape
        self.augment = augment
        self.indexes = np.arange(len(self.file_paths))
        
        # Define augmentation pipeline
        self.augmenter = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
        ])
        
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.file_paths) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Get batch indexes
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.file_paths))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        # Generate data
        X, y = self._generate_batch(batch_indexes)
        return X, y
    
    def on_epoch_end(self):
        """Shuffle indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _generate_batch(self, batch_indexes):
        """Generate batch data from file paths"""
        X_batch = []
        y_batch = []
        
        for idx in batch_indexes:
            file_path = self.file_paths[idx]
            label = self.labels[idx]
            
            # Extract mel spectrogram
            # Note: We apply augmentation on raw audio BEFORE mel extraction if augment=True
            # But extract_mel_spectrogram handles loading internally.
            # So we need to modify this slightly or load audio here first.
            
            # Better approach: Load audio -> Augment -> Extract Mel
            try:
                # We need to use librosa to load to get raw audio
                import librosa
                y, sr = librosa.load(file_path, sr=16000, duration=5)
                
                # Apply augmentation if enabled
                if self.augment:
                    y = self.augmenter(samples=y, sample_rate=sr)
                
                # Now compute mel spectrogram manually or use helper (helper loads file again)
                # To reuse helper logic, we'll just pass the loaded audio 'y' if we refactor helper.
                # But helper expects file_path. Let's refactor helper or just do it here.
                # For simplicity, let's use the helper but we need to pass 'y' to it.
                # Since helper doesn't accept 'y', let's just do the mel extraction here directly
                # to avoid double loading.
                
                # Apply noise reduction (always good)
                from src.preprocess import reduce_noise
                y = reduce_noise(y, sr=sr)
                
                # Compute Mel spectrogram
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                
                # Normalize
                mel = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                mel = None
            
            if mel is not None:
                # Ensure consistent shape
                if mel.shape[1] != self.target_shape[1]:
                    if mel.shape[1] > self.target_shape[1]:
                        mel = mel[:, :self.target_shape[1]]
                    else:
                        mel = np.pad(mel, ((0, 0), (0, self.target_shape[1] - mel.shape[1])), mode='constant')
                
                X_batch.append(mel)
                y_batch.append(label)
        
        # Convert to numpy arrays
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        
        # Add channel dimension
        X_batch = X_batch[..., np.newaxis]
        
        return X_batch, y_batch


def load_data(data_dir):
    """
    Loads file paths and labels from the directory. 
    Supports two modes:
    1. 'dummy' folder with filenames like sample_{id}_{label}.wav
    2. 'dataset' folder with 'real' and 'fake' subdirectories.
    """
    files = []
    labels = []
    
    # Check for real/fake structure
    real_dir = os.path.join(data_dir, 'real')
    fake_dir = os.path.join(data_dir, 'fake')
    
    if os.path.exists(real_dir) and os.path.exists(fake_dir):
        print(f"Loading data from {data_dir} (Real/Fake structure)...")
        # Load Real (Label 0)
        for f in os.listdir(real_dir):
            if f.lower().endswith(('.wav', '.mp3', '.flac')):
                files.append(os.path.join(real_dir, f))
                labels.append(0)
                
        # Load Fake (Label 1)
        for f in os.listdir(fake_dir):
            if f.lower().endswith(('.wav', '.mp3', '.flac')):
                files.append(os.path.join(fake_dir, f))
                labels.append(1)
    else:
        # Fallback to flat directory (dummy data)
        print(f"Loading data from {data_dir} (Flat structure)...")
        for f in os.listdir(data_dir):
            if f.endswith('.wav'):
                path = os.path.join(data_dir, f)
                files.append(path)
                # Filename format: sample_{id}_{label}.wav
                try:
                    label = int(f.split('_')[-1].split('.')[0])
                    labels.append(label)
                except:
                    pass
    
    print(f"Found {len(files)} audio files ({labels.count(0)} real, {labels.count(1)} fake)")
    return files, labels


def plot_history(history, save_path='models/training_history.png'):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path='models/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def train_pipeline():
    """
    Optimized training pipeline for large datasets (25GB+).
    Uses data generators for memory efficiency and advanced callbacks.
    """
    # 1. Create/Load Data Paths
    data_path = 'data/dataset'
    if not os.path.exists(data_path) or not os.path.exists(os.path.join(data_path, 'real')):
        print("Real dataset not found. Generating dummy data...")
        create_dummy_dataset(num_samples=50, save_dir='data/dummy')
        data_path = 'data/dummy'
    
    files, labels = load_data(data_path)
    
    if len(files) == 0:
        print("No audio files found in dataset! Switching to dummy data generation...")
        create_dummy_dataset(num_samples=50, save_dir='data/dummy')
        data_path = 'data/dummy'
        files, labels = load_data(data_path)
    
    # 2. Split data (only file paths, not actual data)
    X_train_files, X_test_files, y_train, y_test = train_test_split(
        files, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(X_train_files)} samples")
    print(f"  Testing: {len(X_test_files)} samples")
    
    # 3. Create data generators (memory-efficient for large datasets)
    train_generator = AudioDataGenerator(
        X_train_files, y_train, 
        batch_size=16,  # Optimized for 8GB VRAM
        shuffle=True,
        augment=True  # Enable data augmentation for training
    )
    
    test_generator = AudioDataGenerator(
        X_test_files, y_test,
        batch_size=16,
        shuffle=False,
        augment=False  # No augmentation for validation
    )
    
    # 4. Initialize Model
    detector = DeepfakeDetector(input_shape=(128, 157, 1))
    detector.summary()
    
    # 5. Setup Callbacks
    os.makedirs('models', exist_ok=True)
    
    # Generate timestamp for unique model filename
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_path = f'models/best_model_{timestamp}.keras'
    
    print(f"Model will be saved to: {model_path}")
    
    # Save best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        model_path, 
        monitor='val_accuracy', 
        save_best_only=True, 
        verbose=1,
        mode='max'
    )
    
    # Stop training when validation accuracy stops improving
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,  # Stop if no improvement for 5 epochs
        verbose=1,
        mode='max',
        restore_best_weights=True
    )
    
    # Reduce learning rate when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduce LR by half
        patience=3,  # Wait 3 epochs before reducing
        verbose=1,
        min_lr=1e-7
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # 6. Train with data generators
    print("\n" + "="*50)
    print("Starting training with optimized configuration:")
    print(f"  Batch size: 16")
    print(f"  Max epochs: 30 (with early stopping)")
    print(f"  Memory: Using data generators (no RAM overflow)")
    print("="*50 + "\n")
    
    history = detector.model.fit(
        train_generator,
        epochs=30,  # Increased for large dataset, early stopping will prevent overfitting
        validation_data=test_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. Evaluate & Save Artifacts
    print("\n" + "="*50)
    print("Training complete! Generating evaluation metrics...")
    print("="*50 + "\n")
    
    plot_history(history)
    
    # Predict on test set
    print("Generating predictions on test set...")
    y_pred_prob = detector.model.predict(test_generator)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    print("\n" + "="*50)
    print("All artifacts saved in 'models/' directory:")
    print(f"  - {os.path.basename(model_path)} (trained model)")
    print("  - training_history.png (accuracy/loss plots)")
    print("  - confusion_matrix.png (confusion matrix)")
    print("="*50)

if __name__ == "__main__":
    train_pipeline()
