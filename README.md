# Deepfake Audio Detection System
**Developed by: Panga Pavan Kalyan**

A deep learning system for detecting AI-generated (deepfake) audio using CNN+BiLSTM architecture with explainable AI features.

## Features

- **High Accuracy Detection**: Detects deepfake audio with 99%+ accuracy using CNN+BiLSTM.
- **Micro-Temporal Analysis**: Pinpoints exactly *when* the audio is fake (e.g., "Fake from 02:45 to 05:10").
- **Forensic Dashboard**: Professional "Cyber-Blue" UI with Spectrograms, Frequency Radar, and XAI Heatmaps.
- **Speaker Verification**: Enroll voice profiles to verify identity matching (Bio-metric + Deepfake Defense).
- **Multiple Input Methods**: 
    - 🎤 Live Recording
    - 📂 File Upload
    - 🧪 One-Click Test Samples (Real, Fake, and Mixed)
- **Mixed Audio & Splicing Detection**:
    - **Smart Verdicts**: Distinguishes between "Strict" (Live) and "Suspicious" (Files) to catch spliced audio.
    - **Segment Breakdown**: Lists exact timestamps: "Real: 0s-4s, Fake: 4s-5s".
    - **Splicing XAI**: Detects "Insertion Attacks" (e.g., adding words to a sentence).
- **Comprehensive Reporting**: Generate detailed PDF Forensic Reports with visualizations and Verdict-specific styling.

## Quick Start

### Prerequisites
- **Python 3.10** (Required for TensorFlow 2.10 GPU compatibility).
- **CUDA 11.2 & cuDNN 8.1** (Recommended for GPU acceleration).
- **8GB+ RAM** (16GB recommended).

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DeepfakeProject
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Launch Dashboard

```bash
streamlit run app.py
```
Access the interface at `http://localhost:8501`.

**Quick Test Features**:
Navigate to the **TEST SAMPLE** tab to instantly load:
- ✅ **Random Real**: Valid human speech sample.
- ⚠️ **Random Fake**: Deepfake sample from ASVspoof.
- 🔄 **Random Mixed**: Complex sample containing both real and fake segments (for Temporal Analysis testing).

### Training a New Model

1. Prepare your dataset in the following structure:
```
data/dataset/
├── train/
│   ├── real/  # Real audio files (.wav, .mp3, .flac)
│   └── fake/  # Deepfake audio files
└── dev/       # Validation set (same structure)
```

2. Open and run `Deepfake_Detection_Complete.ipynb` in Jupyter:
```bash
jupyter notebook Deepfake_Detection_Complete.ipynb
```

The notebook handles:
- Data loading with memory-efficient generators
- Model training with early stopping (patience=3)
- Automatic model saving to `models/model_YYYYMMDD_HHMMSS/`
- Metrics and plot generation

## Project Structure

```
DeepfakeProject/
├── app.py                              # Main Forensic Dashboard (Streamlit)
├── Deepfake_Detection_Complete.ipynb   # Main Training Notebook
├── requirements.txt                    # Project Dependencies
├── backups/                            # Backup of critical files
├── data/
│   ├── dataset/                        # ASVspoof 2019 Dataset
│   └── speaker_profiles/               # Enrolled Speaker Database (Pickle files)
├── models/                             # Trained Models (Timestamped)
├── src/                                # Core Logic Modules
│   ├── explainer.py                    # XAI / Grad-CAM Logic
│   ├── speaker_recognition.py          # Voice ID (Resemblyzer)
│   ├── temporal_analyzer.py            # Sliding Window Detection Logic
│   ├── temporal_visualizer.py          # Timeline Plotting
│   └── report_generator.py             # PDF Reporting
└── scripts/                            # Utility Scripts
```

## Dataset

The model works on a diverse dataset constructed from two primary sources to ensure robust generalization:

### 1. ASVspoof 2019 LA (Logical Access)
- **Primary Source** for Deepfake attacks.
- Contains high-quality synthesized speech using various TTS and VC algorithms.
- **Reference**: [ASVspoof Challenge](https://www.asvspoof.org)

### 2. LibriSpeech (Clean)
- **Primary Source** for Real human speech.
- Used to augment the "Real" class with diverse speakers and accents.
- Ensures the model doesn't overfit to specific recording conditions of ASVspoof.
- **Reference**: [OpenSLR](https://www.openslr.org/12)

```bash
python prepare_dataset.py
```

This script automatically:
- Extracts the dataset
- Organizes files into train/dev/eval splits
- Separates real and fake audio

## Model Architecture

**Hybrid CNN + BiLSTM**

- **Input**: Mel spectrogram (128 × 157 × 1) representing 5 seconds of audio
- **CNN Layers**: 3 blocks (32, 64, 128 filters) for feature extraction
- **BiLSTM Layers**: 2 blocks (64, 32 units) for temporal pattern analysis
- **Output**: Binary classification (Real/Fake) with sigmoid activation
- **Total Parameters**: ~1.2M

### Training Configuration

- **Optimizer**: Adam
- **Loss**: Binary Crossentropy
- **Batch Size**: 16
- **Early Stopping**: Patience = 3 epochs
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

## Using the Web Interface

1. **Start the app**: `python -m streamlit run app.py`

2. **Choose input method**:
   - Upload audio file (WAV/MP3)
   - Record live audio

3. **Analyze**:
   - Click "Analyze Audio"
   - View real-time prediction and confidence score
   - Check waveform and mel spectrogram

4. **Enable XAI features** (sidebar):
   - ✅ Show Detailed Explanation
   - ✅ Show Grad-CAM Heatmap
   - ✅ Generate PDF Report

## Performance Metrics

The app automatically displays metrics from the latest trained model:
- **Confusion Matrix**: True/False positives and negatives
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score

## Generating Metrics for Saved Models

If you have a trained model without metrics:

```bash
python generate_presentation_plots.py
```

This will:
- Find the latest model in `models/`
- Evaluate on the validation set
- Generate confusion matrix and performance plots
- Save results in the model's folder

## Technical Details

### Audio Preprocessing

1. Load audio at 16kHz, 5-second duration
2. Apply noise reduction (80% prop_decrease)
3. Extract mel spectrogram (128 mel bands)
4. Convert to dB scale and normalize to [0, 1]

### Memory Optimization

- **Data Generators**: Load audio on-the-fly in batches
- **Batch Size 16**: Optimized for 8GB VRAM
- **Timestamped Models**: Each training run saved separately

### Explainable AI

- **Grad-CAM**: Highlights frequency-time regions that influenced the decision
- **Region Analysis**: Identifies suspicious time ranges and frequency bands
- **Confidence-Based Explanations**: Contextual text based on prediction score

## Troubleshooting

### GPU Not Detected

Ensure CUDA 11.2 and cuDNN 8.1 are installed and in PATH:
```bash
python verify_gpu_final.py
```

### Out of Memory

- Reduce batch size to 8 in the notebook
- Close other applications
- Use CPU-only mode (slower but works)

### Model Not Loading in App

- Check that `models/model_*/model.keras` exists
- Verify the model was trained successfully
- Check console for error messages

## Documentation

- **README.md** (this file): Quick start and usage
- **DOCUMENTATION.md**: Detailed technical documentation

## Project Info

**Academic Project: Deepfake Audio Detection System**

**Objective**: To develop a robust, forensic-grade application capable of detecting AI-generated synthesized speech with high precision and explainability.

**Core Technologies**:
- **Deep Learning**: TensorFlow/Keras (CNN + BiLSTM)
- **Signal Processing**: Librosa (Mel Spectrograms, Micro-Temporal Analysis)
- **Bio-Metrics**: Resemblyzer (Deep Speaker Embeddings)
- **Visualization**: Matplotlib, Streamlit, Grad-CAM

**Maintainer**: [Your Name/Team]

