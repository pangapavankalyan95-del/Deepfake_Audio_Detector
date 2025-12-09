# Technical Documentation

Detailed technical information about the deepfake audio detection system.

## Table of Contents
1. [Model Architecture](#model-architecture)
2. [Audio Processing](#audio-processing)
3. [Training Details](#training-details)
4. [Grad-CAM Implementation](#grad-cam-implementation)
5. [Performance Notes](#performance-notes)
6. [Code Reference](#code-reference)

---
## Model Architecture

### Overall Structure

The model has three main parts:

1. CNN layers (3 blocks) - extract features from mel spectrograms
2. BiLSTM layers (2 blocks) - analyze temporal patterns
3. Dense layers - final classification

### Layer-by-Layer Breakdown

**Input Layer**
- Shape: (128, 157, 1)
- This is a mel spectrogram: 128 frequency bands, 157 time frames, 1 channel

**CNN Block 1**
- Conv2D with 32 filters (3x3 kernel)
- MaxPooling (2x2)
- BatchNormalization
- Purpose: Extract basic features

**CNN Block 2**
- Conv2D with 64 filters (3x3 kernel)
- MaxPooling (2x2)
- BatchNormalization
- Purpose: Extract more complex features

**CNN Block 3**
- Conv2D with 128 filters (3x3 kernel)
- MaxPooling (2x3)
- BatchNormalization
- Purpose: High-level feature extraction

**Reshape Layer**
- Converts CNN output to sequence format for BiLSTM
- Uses Permute and Reshape operations

**BiLSTM Block 1**
- 64 units, bidirectional
- Returns sequences for next layer
- Processes temporal patterns

**BiLSTM Block 2**
- 32 units, bidirectional
- Returns final state only
- Combines temporal information

**Dense Layers**
- Dense(64) with ReLU
- Dropout(0.3) to prevent overfitting
- Dense(1) with Sigmoid for binary output

**Total Parameters**: About 344,000

### Why This Design?

I chose CNN+BiLSTM because:
- CNNs are good at finding patterns in images (mel spectrograms are basically images)
- BiLSTMs can understand how things change over time
- Together they handle both "what" patterns exist and "when" they occur

The specific numbers (32, 64, 128 filters) are pretty standard for this type of problem. I tried a few different configurations and this worked well.

---

## Audio Processing

### Converting Audio to Mel Spectrograms

Here's what happens when you load an audio file:

**Step 1: Load the audio**
```python
librosa.load(file_path, sr=16000, duration=5)
```
- Sample rate: 16000 Hz (standard for speech)
- Duration: 5 seconds (I pad or trim to this length)

**Step 2: Noise reduction**
```python
noisereduce.reduce_noise(audio, sr=16000, prop_decrease=0.8)
```
- Removes background noise
- Set to 80% reduction (not too aggressive)
- Helps with real-world recordings

**Step 3: Create mel spectrogram**
```python
librosa.feature.melspectrogram(audio, sr=16000, n_mels=128)
```
- 128 mel bands (frequency resolution)
- Results in 157 time frames for 5 seconds

**Step 4: Convert to decibels**
```python
librosa.power_to_db(mel_spec)
```
- Makes the values easier to work with

**Step 5: Normalize**
```python
(mel_spec - min) / (max - min)
```
- Scales everything to 0-1 range
- Helps the model train better

### Why Mel Spectrograms?

Mel spectrograms show:
- Frequency content (Y-axis)
- How it changes over time (X-axis)
- Energy levels (color/brightness)

They're better than raw audio because:
- Much smaller (128x157 vs 80,000 samples)
- Captures what humans hear
- Works well with CNNs

---

## Training Details

### Memory-Efficient Data Loading

To handle the large combined dataset (ASVspoof + LibriSpeech > 30GB), a custom data generator was implemented:

```python
class AudioDataGenerator:
    # Only stores file paths, not actual audio
    # Loads and processes audio on-the-fly in batches
```

This approach ensures:
- **Low Memory Footprint**: Only loads `BATCH_SIZE` (16) files at a time.
- **Scalability**: Can train on datasets of any size without RAM bottlenecks.
- **Real-Time Augmentation**: Applies noise/effects during loading (if enabled).

### Training Configuration

**Batch size: 16**
- Tested 8, 16, 32
- 16 works best for 8GB VRAM
- Faster than 8, doesn't crash like 32

**Epochs: Up to 30**
- Usually stops around 10-15 due to early stopping
- **Early stopping patience: 3 epochs** (updated from 7)
- Monitors validation accuracy

**Optimizer: Adam**
- Default learning rate (0.001)
- Works well for this problem

**Loss: Binary Crossentropy**
- Standard for binary classification

### Callbacks Used

**ModelCheckpoint**
- Saves the best model during training
- Monitors validation accuracy
- **Saves to timestamped folder**: `models/model_YYYYMMDD_HHMMSS/model.keras`
- Each training run gets its own folder with metrics

**EarlyStopping**
- Stops if validation accuracy doesn't improve for **3 epochs**
- Prevents overfitting
- Restores the best weights automatically

**ReduceLROnPlateau**
- Cuts learning rate in half if loss stops improving
- Patience: 3 epochs
- Helps with fine-tuning

### Data Split

- 80% training
- 20% validation
- Random split with shuffling

### Training Workflow

1. Run `Deepfake_Detection_Complete.ipynb`
2. Model trains with callbacks
3. Best model saved to `models/model_YYYYMMDD_HHMMSS/`
4. Metrics and plots generated automatically
5. Results include:
   - `model.keras` - Trained model
   - `metrics.json` - Performance metrics
   - `training_history.png` - Accuracy/loss curves
   - `confusion_matrix.png` - Confusion matrix
   - `performance_metrics.png` - Bar chart of metrics

---

## Grad-CAM Implementation

### How It Works

Grad-CAM shows which parts of the input were important for the decision. Here's my implementation:

1. **Get the prediction**
   - Run the audio through the model
   - Get the output score

2. **Calculate gradients**
   - Find how much each part of the last CNN layer affected the output
   - This tells us which features were important

3. **Create heatmap**
   - Combine the gradients with the feature maps
   - Resize to match the input size
   - Normalize to 0-1 range

4. **Overlay on spectrogram**
   - Use a color map (red = important, blue = not important)
   - Set transparency to 50% so you can see both

### Region Analysis

After generating the heatmap, I analyze it to find:

**Time ranges**
- Which seconds of audio were suspicious
- Example: "2.3s to 3.1s"

**Frequency bands**
- Which frequencies had problems
- Example: "mel bands 45-78"

**Coverage**
- What percentage of the audio was flagged
- Helps determine confidence

### Explanation Generation

I use templates based on the analysis:

```python
if fake and confidence > 0.8:
    "Very high confidence of manipulation"
elif fake and confidence > 0.6:
    "Clear signs of manipulation"
else:
    "Possible manipulation detected"
```

This is more reliable than trying to generate explanations automatically.

### PDF Reports

The PDF includes:
- Metadata (date, filename, model version)
- Detection result (colored based on fake/real)
- Explanation text
- Grad-CAM heatmap image
- Disclaimer

I'm using ReportLab for PDF generation. It took a while to get the formatting right.

---

## Micro-Temporal Analysis

The system now goes beyond binary classification by analyzing audio in a sliding window to detect *temporally localized* deepfakes (e.g., a 2-second fake segment inside a 1-minute real recording).

### Algorithm: Sliding Window & Micro-Scanning
1.  **Window Size**: 5.0 seconds (matches training input size).
2.  **Overlap**: 50% (2.5 seconds).
3.  **Ping-Pong Tiling**: When analyzing short segments (<5s), the audio is mirrored (`[chunk, chunk[::-1]]`) before tiling. This ensures signal continuity at boundaries, preventing "click" artifacts that cause false positives.
4.  **Smart Noise Reduction**: During micro-scans, a lighter Noise Reduction (0.1 strength) is applied to remove hiss without scrubbing subtle deepfake artifacts.
5.  **Aggregation**: Predictions for overlapping regions are averaged (smoothed).
6.  **Thresholding**: Regions consistently above 0.3 are flagged as fake.

### Verdict Logic
The system uses a smart decision tree based on input type:
*   **Live Recording**: Strict Binary Verdict (Real/Fake). Ignores ambiguous signals to prevent false alarms from background noise.
*   **File/Upload**: Triggers "Mixed" analysis.
    *   **Fake Ratio > 95%**: Verdict "FAKE".
    *   **Fake Ratio > 0%**: Verdict "SUSPICIOUS / MIXED".
    *   **Fake Ratio = 0%**: Verdict "REAL".

### Visualization Colors
- **Dark Green**: High Confidence Real (Score ~0.0)
- **Light Green/Beige**: Low Confidence Real (Score ~0.3-0.4). Often indicates background noise but safe content.
- **Red**: Fake (Score > 0.5)

This allows the **Timeline Visualization** in the dashboard to show a red/green curve over time, helping identifying mixed attacks.

## Speaker Verification

To prevent "Identity Theft" deepfakes, the system includes a bio-metric layer using **Resemblyzer**.

1. **Enrollment**: Analyzes 3-5 real samples of a user to create a high-dimensional voice embedding.
2. **Storage**: Saves profiles as encrypted Pickle files in `data/speaker_profiles`.
3. **Verification**: During analysis, comparing the input voice against the enrolled database.
   - **Confidence > 75%**: Confirmed Identity.
   - **Confidence < 75%**: Unknown Speaker or Deepfake Voice Clone.


## Performance Notes

### Processing Times

Based on my testing:
- Loading audio: ~0.1s
- Noise reduction: ~0.5s
- Mel spectrogram: ~0.2s
- Model prediction: ~0.5s
- Grad-CAM: ~1.5s (optional)
- PDF generation: ~1s (optional)

Total: 1-2 seconds without XAI, 3-4 seconds with everything enabled.

### Memory Usage

- Model: ~50MB RAM, ~200MB VRAM
- One batch (16 samples): ~100MB RAM, ~500MB VRAM
- Grad-CAM: +50MB RAM, +100MB VRAM

Total VRAM usage stays under 1GB, which is why batch size 16 works well.

### Optimization Tips

If you're running out of memory:
- Reduce batch size to 8
- Disable Grad-CAM when not needed
- Close other applications

For faster training:
- Use a GPU if available
- Increase batch size if you have more VRAM
- Consider mixed precision training (didn't implement this yet)

---

## Code Reference

### Main Files

**Deepfake_Detection_Complete.ipynb**
- Complete training pipeline in Jupyter notebook
- Includes data loading, model building, training, and evaluation
- Automatically generates metrics and plots

**app.py**
- Streamlit web interface
- Handles file upload and live recording
- Integrates with explainer for XAI features
- Automatically loads latest model from timestamped folders

**generate_presentation_plots.py**
- Standalone script to generate metrics for saved models
- Evaluates model on validation set
- Creates confusion matrix and performance plots
- Saves results to model folder

**prepare_dataset.py**
- Organizes ASVspoof 2019 dataset
- Extracts and splits into train/dev/eval
- Separates real and fake audio

### Key Functions

**Core Logic (src/)**

`TemporalAnalyzer.analyze_temporal(audio_path)`:
- Performs the sliding window analysis.
- Returns: List of fake segments (start_time, end_time, confidence).

`SpeakerRecognizer.identify_speaker(audio_path)`:
- Extracts voice embedding from input.
- Computes Cosine Similarity against `data/speaker_profiles`.
- Returns: Best match name and confidence score.

`ForensicReportGenerator.generate_report(...)`:
- Compiles Mel Spectrograms, Radar Charts, and Temporal Timelines into a professional PDF.


**Audio Processing (in notebook and app.py)**

`load_audio(file_path, sr=16000, duration=5)`
- Loads audio and pads/truncates to fixed duration
- Returns: numpy array

`reduce_noise(audio, sr=16000)`
- Applies noise reduction to audio
- Returns: cleaned audio array

`extract_mel_spectrogram(file_path, sr=16000, n_mels=128, duration=5, apply_noise_reduction=True)`
- Loads audio and creates mel spectrogram
- Returns: numpy array (128, 157)

**Model (in notebook)**

`build_model(input_shape=(128, 157, 1))`
- Creates the CNN+BiLSTM model
- Returns: compiled Keras model

**Data Generator (in notebook)**

`AudioDataGenerator(file_paths, labels, batch_size=16, shuffle=True, augment=False)`
- Memory-efficient data loading
- Loads audio on-the-fly
- Optional augmentation for training

**Explainer (src/explainer.py)**

`DeepfakeExplainer(model)`
- Initialize with trained model

`generate_gradcam(mel_spectrogram, prediction_score)`
- Creates Grad-CAM heatmap
- Returns: heatmap array

`generate_explanation(prediction_score, analysis_results)`
- Creates text explanation
- Returns: formatted string

`generate_pdf_report(...)`
- Creates PDF with all analysis
- Returns: path to PDF file

### Configuration Parameters

All in the respective files, but here are the key ones:

**Audio:**
- SAMPLE_RATE = 16000
- DURATION = 5
- N_MELS = 128

**Training:**
- BATCH_SIZE = 16
- MAX_EPOCHS = 30
- EARLY_STOPPING_PATIENCE = 3
- LEARNING_RATE = 0.001 (Adam default)

**Noise Reduction:**
- PROP_DECREASE = 0.8 (80% reduction)

**Grad-CAM:**
- HEATMAP_ALPHA = 0.5 (50% transparency)
- THRESHOLD = 0.6 (for region detection)

**Detection:**
- THRESHOLD = 0.5 (fixed in app)

---

## Common Issues I Ran Into

**Out of memory during training**
- Solution: Reduced batch size and implemented data generators

**Grad-CAM showing weird patterns**
- Solution: Make sure to normalize the heatmap properly

**Model not learning**
- Solution: Check that data is properly labeled (0 for real, 1 for fake)

**Slow training**
- Solution: Use data generators, they're much faster than loading everything

**PDF generation failing**
- Solution: Make sure the heatmap figure is created before passing to PDF function

---

## Things I Might Add Later

- Support for longer audio files (currently limited to 5 seconds)
- More data augmentation (time stretching, pitch shifting)
- Ensemble of multiple models
- Better handling of music vs speech
- Real-time streaming analysis

---

For basic usage, see the main [README.md](README.md)
