# Deepfake Audio Detection System

A B.Tech final year project for detecting AI-generated (deepfake) audio using deep learning.

## What This Project Does

This system can tell if an audio file is real or AI-generated (deepfake). I built it using a CNN+BiLSTM neural network that analyzes mel spectrograms of audio files.

Main features:
- Detects fake audio with good accuracy
- Shows you WHERE in the audio it found suspicious patterns (using Grad-CAM heatmaps)
- Explains WHY it thinks the audio is fake or real
- Can generate PDF reports for documentation
- Works with uploaded files or live microphone recording

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training the Model
```bash
python -m src.train
```
Note: If you don't have a dataset, it will create dummy data for testing. For real results, you need to download an actual deepfake dataset (see below).

### Running the Web Interface
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

## Project Structure

```
├── data/dataset/       # Put your audio files here (real/ and fake/ folders)
├── models/             # Trained models get saved here
├── src/
│   ├── preprocess.py  # Handles audio loading and mel spectrogram extraction
│   ├── model.py       # The CNN+BiLSTM model architecture
│   ├── train.py       # Training script with all the optimizations
│   └── explainer.py   # Grad-CAM and PDF report generation
├── app.py             # Streamlit web interface
└── requirements.txt   # All the packages you need
```

## How to Use

1. Start the web app (see above)
2. Either upload an audio file or record using your microphone
3. Click "Analyze Audio"
4. Check the sidebar to enable:
   - Detailed explanation (tells you why it's fake/real)
   - Grad-CAM heatmap (shows where it found problems)
   - PDF report (downloads a full report)

## Getting a Dataset

To train the model properly, you need real and fake audio samples. I recommend:

**ASVspoof 2019** (what I used)
- Download: https://datashare.ed.ac.uk/handle/10283/3336
- Get the "Logical Access (LA)" part
- Size: About 25GB
- Has both real and synthesized speech

Put the files in this structure:
```
data/dataset/
├── real/  # Real audio files go here
└── fake/  # Fake/deepfake audio files go here
```

## The Model

I'm using a hybrid CNN+BiLSTM architecture:

- **CNN part**: Extracts features from the mel spectrogram (like finding patterns in the frequency data)
- **BiLSTM part**: Looks at how these patterns change over time
- **Output**: A score from 0 to 1 (0 = real, 1 = fake)

Input is a mel spectrogram with 128 frequency bands and 157 time steps (represents 5 seconds of audio).

## Training Optimizations

Since I'm working with a 25GB dataset on a laptop with 8GB VRAM, I had to optimize things:

- **Data generators**: Loads audio files in batches instead of all at once (saves RAM)
- **Batch size of 16**: Good balance for 8GB VRAM
- **Early stopping**: Stops training if accuracy isn't improving
- **Learning rate scheduling**: Automatically reduces learning rate when needed
- **Noise reduction**: Cleans up audio before processing

Models are saved with timestamps so you don't lose previous training runs.

## Explainable AI Features

This was the interesting part - making the model explain its decisions:

**Grad-CAM Heatmaps**
- Shows which parts of the audio made the model suspicious
- Red areas = detected problems
- Blue areas = looks normal

**Text Explanations**
- Tells you the time range where it found issues (like "2.3 to 3.1 seconds")
- Shows which frequency ranges had problems
- Gives an interpretation based on confidence level

**PDF Reports**
- Combines everything into a downloadable PDF
- Useful for documentation or showing results

## Technical Details

For detailed technical documentation, see [DOCUMENTATION.md](DOCUMENTATION.md)

## Notes

- The model needs to be trained on real data to work properly. The dummy data is just for testing the pipeline.
- Processing time is about 1-2 seconds per audio file (3-4 seconds if you enable all the XAI features)
- Works best with clear speech audio, might struggle with music or very noisy recordings

## Project Info

B.Tech Final Year Project  
Topics: Deep Learning, Audio Processing, Explainable AI
