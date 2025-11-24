import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tensorflow.keras.models import load_model
from audio_recorder_streamlit import audio_recorder
import soundfile as sf
import tempfile

from src.preprocess import extract_mel_spectrogram
from src.explainer import DeepfakeExplainer

# Page Config
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        border-color: #ff6b6b;
    }
    h1 {
        color: #1f2937;
        text-align: center;
        padding-bottom: 20px;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector_model():
    model_dir = 'models'
    if not os.path.exists(model_dir):
        return None
        
    # Find all keras files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
    
    if not model_files:
        return None
        
    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    
    latest_model = os.path.join(model_dir, model_files[0])
    return load_model(latest_model)

def predict_audio(model, file_path):
    # Extract features
    mel_spec = extract_mel_spectrogram(file_path)
    
    if mel_spec is None:
        return None, None
    
    # Preprocess for model (shape: 1, 128, 157, 1)
    # Resize/Pad to match training shape (157 time steps)
    target_width = 157
    if mel_spec.shape[1] > target_width:
        mel_spec = mel_spec[:, :target_width]
    else:
        mel_spec = np.pad(mel_spec, ((0,0), (0, target_width - mel_spec.shape[1])), mode='constant')
        
    # Add batch and channel dims
    input_data = mel_spec[np.newaxis, ..., np.newaxis]
    
    # Predict
    prediction = model.predict(input_data)
    return prediction[0][0], mel_spec

def main():
    st.title("🎙️ Deepfake Audio Detection System")
    st.markdown("### CNN + BiLSTM Architecture")
    
    model = load_detector_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please run the training script first.")
        st.info("Run `python src/train.py` to generate the model.")
        return

    # Sidebar
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Choose Input Method:", ["Upload Audio File", "Live Recording"])
    
    st.sidebar.markdown("---")
    st.sidebar.title("XAI Options")
    show_explanation = st.sidebar.checkbox("Show Detailed Explanation", value=True)
    show_heatmap = st.sidebar.checkbox("Show Grad-CAM Heatmap", value=True)
    generate_pdf = st.sidebar.checkbox("Generate PDF Report", value=False)
    
    st.sidebar.markdown("---")
    st.sidebar.info("This system uses a Deep Learning model to analyze audio features (Mel Spectrograms) and detect potential deepfakes.")

    audio_file_path = None

    if option == "Upload Audio File":
        st.subheader("📂 Upload an Audio File")
        uploaded_file = st.file_uploader("Choose a WAV or MP3 file", type=['wav', 'mp3'])
        
        if uploaded_file is not None:
            # Save temp file
            with open("temp_upload.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            audio_file_path = "temp_upload.wav"
            st.audio(audio_file_path)

    elif option == "Live Recording":
        st.subheader("🔴 Live Audio Recording")
        st.write("Click the microphone to start recording.")
        
        audio = audio_recorder(text="Click to record", icon_size="2x")
        
        if audio is not None and len(audio) > 0:
            st.audio(audio)
            
            # Save recording
            with open("temp_record.wav", "wb") as f:
                f.write(audio)
            audio_file_path = "temp_record.wav"

    # Analysis Section
    if audio_file_path:
        # Display Waveform
        st.subheader("🔊 Audio Waveform")
        y, sr = librosa.load(audio_file_path, sr=16000)
        fig_wave, ax_wave = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax_wave, color='blue', alpha=0.5)
        ax_wave.set_title("Waveform")
        st.pyplot(fig_wave)

        if st.button("🔍 Analyze Audio"):
            with st.spinner("Analyzing audio patterns..."):
                score, mel_spec = predict_audio(model, audio_file_path)
                
                if score is not None:
                    st.markdown("### Analysis Result")
                    
                    # Confidence Progress Bar
                    # Score is probability of being FAKE (1.0)
                    st.progress(float(score))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Threshold 0.5
                        if score > 0.5:
                            st.error(f"🚨 **FAKE DETECTED**")
                            st.metric("Confidence (Fake)", f"{score*100:.2f}%")
                        else:
                            st.success(f"✅ **REAL AUDIO**")
                            st.metric("Confidence (Real)", f"{(1-score)*100:.2f}%")
                            
                    with col2:
                        st.markdown("### Mel Spectrogram")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        img = librosa.display.specshow(mel_spec, x_axis='time', y_axis='mel', sr=16000, ax=ax)
                        fig.colorbar(img, ax=ax, format='%+2.0f dB')
                        ax.set_title('Mel-frequency spectrogram')
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # XAI Features
                    if show_heatmap or show_explanation or generate_pdf:
                        st.markdown("---")
                        st.markdown("### 🔍 Explainable AI Analysis")
                        
                        with st.spinner("Generating explanations..."):
                            # Initialize explainer
                            explainer = DeepfakeExplainer(model)
                            
                            # Generate Grad-CAM heatmap
                            input_data = mel_spec[np.newaxis, ..., np.newaxis]
                            heatmap = explainer.generate_gradcam(input_data, score)
                            
                            # Analyze heatmap regions
                            analysis_results = explainer.analyze_heatmap_regions(heatmap, mel_spec)
                            
                            # Generate explanation
                            explanation = explainer.generate_explanation(score, analysis_results)
                            
                            if show_heatmap:
                                st.markdown("#### Grad-CAM Heatmap")
                                st.info("Red regions indicate areas that contributed most to the detection decision.")
                                heatmap_fig = explainer.overlay_heatmap(mel_spec, heatmap, alpha=0.5)
                                st.pyplot(heatmap_fig)
                                plt.close(heatmap_fig)
                            
                            if show_explanation:
                                st.markdown("#### Detailed Explanation")
                                st.markdown(explanation)
                            
                            if generate_pdf:
                                st.markdown("#### PDF Report")
                                with st.spinner("Generating PDF report..."):
                                    # Create temp PDF file
                                    pdf_path = tempfile.mktemp(suffix=".pdf")
                                    heatmap_fig = explainer.overlay_heatmap(mel_spec, heatmap, alpha=0.5)
                                    
                                    pdf_file = explainer.generate_pdf_report(
                                        audio_file_path,
                                        score,
                                        mel_spec,
                                        heatmap_fig,
                                        explanation,
                                        pdf_path
                                    )
                                    
                                    plt.close(heatmap_fig)
                                    
                                    if pdf_file and os.path.exists(pdf_file):
                                        with open(pdf_file, "rb") as f:
                                            pdf_bytes = f.read()
                                        
                                        st.download_button(
                                            label="📄 Download PDF Report",
                                            data=pdf_bytes,
                                            file_name=f"deepfake_report_{score:.2f}.pdf",
                                            mime="application/pdf"
                                        )
                                        st.success("PDF report generated successfully!")
                                    else:
                                        st.error("Failed to generate PDF report.")
                else:
                    st.error("Error processing audio file.")

    # Show Training Metrics
    st.markdown("---")
    st.subheader("📊 Model Performance Metrics")
    
    col_a, col_b = st.columns(2)
    
    if os.path.exists('models/training_history.png'):
        with col_a:
            st.image('models/training_history.png', caption='Training History (Accuracy & Loss)')
            
    if os.path.exists('models/confusion_matrix.png'):
        with col_b:
            st.image('models/confusion_matrix.png', caption='Confusion Matrix')

if __name__ == "__main__":
    main()
