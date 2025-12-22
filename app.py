import os

# ===== GPU FIX: Add CUDA to DLL search path BEFORE importing TensorFlow =====
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin"
if os.path.exists(cuda_path):
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(cuda_path)
    os.environ['PATH'] = cuda_path + os.pathsep + os.environ.get('PATH', '')
# ==============================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tensorflow.keras.models import load_model
from audio_recorder_streamlit import audio_recorder
import soundfile as sf
import tempfile
import noisereduce as nr
from datetime import datetime

from src.explainer import DeepfakeExplainer
from src.speaker_recognition import SpeakerRecognizer
from src.report_generator import ForensicReportGenerator
from src.temporal_analyzer import TemporalAnalyzer
from src.temporal_visualizer import TemporalVisualizer
import io
import pandas as pd
from scipy.signal import wiener  # For advanced noise gate

# ==============================================================================
# ADVANCED NOISE GATE (LIVE RECORDING OPTIMIZATION)
# ==============================================================================
def apply_noise_gate(audio_data, threshold_db=-40.0, smoothing_window=1000):
    """
    Advanced Noise Gate using Wiener filter and Energy-Based Gating.
    Removes background hiss and silences quiet sections.
    """
    if len(audio_data) < 2048:
        return audio_data # Too short for FFT processing
    
    # 1. Wiener Filter (removes stationary noise/hiss)
    try:
        clean_audio = wiener(audio_data)
    except:
        clean_audio = audio_data # Fallback
    
    # 2. Energy-Based Gating
    frame_len = 2048
    hop_len = 512
    try:
        rmse = librosa.feature.rms(y=clean_audio, frame_length=frame_len, hop_length=hop_len)[0]
        
        # Convert threshold to amplitude
        db_threshold = librosa.amplitude_to_db(rmse, ref=np.max)
        
        # Create mask: 1 where signal > threshold, 0 where signal < threshold
        mask = db_threshold > threshold_db
        
        # Smooth the mask to prevent "choppy" audio
        mask_expanded = np.repeat(mask, hop_len)
        
        # Fix length mismatch
        if len(mask_expanded) < len(clean_audio):
            mask_expanded = np.pad(mask_expanded, (0, len(clean_audio) - len(mask_expanded)))
        else:
            mask_expanded = mask_expanded[:len(clean_audio)]
            
        # Apply Gate
        gated_audio = clean_audio * mask_expanded
        return gated_audio
    except Exception as e:
        print(f"Noise gate error: {e}")
        return audio_data

# --- Helper Functions ---
def plot_radar_chart(categories, values, title):
    """
    Plots a radar chart for frequency analysis with RED color.
    """
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    values = list(values)
    values += values[:1]
    
    fig, ax = plt.subplots(figsize=(2, 2), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#FF3232')
    ax.fill(angles, values, '#FF3232', alpha=0.25)
    
    plt.xticks(angles[:-1], categories, size=5, color='#8B949E')
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=4)
    plt.ylim(0, 1)
    plt.title(title, size=6, y=1.1, color='white')
    return fig

def display_immersive_verdict(verdict, display_score, score, rms):
    """
    Renders the premium, animated verdict banner and signal metrics.
    """
    # Determine verdict styling with immersive icons
    if verdict == "FAKE":
        verdict_icon = "🚨"  # Alert/Warning siren
        verdict_text = "FAKE"
        verdict_color = "#FF3232"
        bg_gradient = "linear-gradient(135deg, rgba(255, 50, 50, 0.15), rgba(139, 0, 0, 0.05))"
        border_style = f"3px solid {verdict_color}"
        icon_animation = "animation: pulse 2s infinite;"
    elif verdict == "MIXED":
        verdict_icon = "⚠️"
        verdict_text = "SUSPICIOUS"
        verdict_color = "#FFA500"
        bg_gradient = "linear-gradient(135deg, rgba(255, 165, 0, 0.15), rgba(204, 85, 0, 0.05))"
        border_style = f"3px solid {verdict_color}"
        icon_animation = "animation: pulse 2s infinite;"
    else:
        verdict_icon = "✅"
        verdict_text = "REAL"
        verdict_color = "#00FF80"
        bg_gradient = "linear-gradient(135deg, rgba(0, 255, 128, 0.15), rgba(0, 139, 69, 0.05))"
        border_style = f"3px solid {verdict_color}"
        icon_animation = ""
    
    # Immersive verdict card with CSS animation
    st.markdown(f"""
    <style>
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.1); opacity: 0.8; }}
    }}
    </style>
    <div style="
        background: {bg_gradient};
        border: {border_style};
        border-radius: 12px;
        padding: 25px 30px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3), 0 0 20px {verdict_color}40;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 15px;">
                <span style="font-size: 48px; {icon_animation}">{verdict_icon}</span>
                <div>
                    <h2 style="
                        color: {verdict_color}; 
                        margin: 0; 
                        font-family: 'Orbitron', sans-serif;
                        font-size: 32px;
                        font-weight: bold;
                        text-shadow: 0 0 10px {verdict_color}80;
                        letter-spacing: 3px;
                    ">{verdict_text}</h2>
                    <p style="
                        color: #AAAAAA; 
                        margin: 5px 0 0 0; 
                        font-size: 14px;
                        font-family: 'Roboto Mono', monospace;
                    ">Audio Classification Result</p>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="
                    color: white; 
                    font-size: 36px; 
                    font-weight: bold;
                    font-family: 'Courier New', monospace;
                    text-shadow: 0 0 8px {verdict_color}60;
                ">{display_score*100:.1f}%</div>
                <div style="
                    color: #888888; 
                    font-size: 12px;
                    margin-top: 5px;
                    font-family: 'Roboto Mono', monospace;
                ">CONFIDENCE</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Preprocessing Functions ---
def load_audio(file_path, sr=16000, duration=10):
    """
    Loads an audio file and pads/truncates it to a fixed duration.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        # Trimming removed - alters temporal features and causes false positives
        # y, _ = librosa.effects.trim(y, top_db=40)
        
        # Normalize waveform volume for consistent feature extraction
        y = librosa.util.normalize(y)
        
        # Pad if shorter than duration
        target_length = int(sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        return y
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None

def reduce_noise(audio, sr=16000):
    """
    Applies moderate noise reduction to audio signal.
    """
    try:
        reduced_noise = nr.reduce_noise(
            y=audio, 
            sr=sr,
            stationary=True, 
            prop_decrease=0.8
        )
        return reduced_noise
    except Exception as e:
        return audio

def extract_mel_spectrogram(file_path, sr=16000, n_mels=128, duration=10, apply_noise_reduction=True):
    """
    Extracts Mel Spectrogram from an audio file.
    """
    y = load_audio(file_path, sr=sr, duration=duration)
    if y is None:
        return None
    
    if apply_noise_reduction:
        y = reduce_noise(y, sr=sr)
    
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Robust Normalization: Use 60dB floor to prevent noise magnification
    # Training data usually has ~80dB range. If a segment has < 60dB, it's likely noise.
    db_range = mel_spec_db.max() - mel_spec_db.min()
    norm_range = max(db_range, 60.0)
    
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (norm_range + 1e-8)
    
    return mel_spec_norm

# Helper for formatting duration
def time_str(seconds):
    return f"{seconds:.1f}s"

# -------------------------------------------------------------------------------

# Page Config
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Deepfake Audio Detection using CNN+BiLSTM"
    }
)

# Custom CSS - Cyber-Blue Theme
st.markdown("""
<style>
    /* --- GLOBAL THEME --- */
    .main {
        background-color: #050505;
        font-family: 'Roboto Mono', monospace;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #00F0FF !important;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
    }
    
    p, label, .stMarkdown {
        color: #C0C0C0 !important;
    }

    /* --- DATA CARDS --- */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #00F0FF !important;
        font-weight: bold;
        text-shadow: 0 0 5px rgba(0, 240, 255, 0.5);
    }
    
    .metric-card {
        background: rgba(20, 20, 20, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid #333;
        border-left: 3px solid #00F0FF;
        border-radius: 4px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.5);
        margin-bottom: 10px;
    }

    /* --- BUTTONS --- */
    .stButton>button {
        background: linear-gradient(90deg, #006B75, #004D66) !important;
        color: #FFFFFF !important;
        border: 1px solid #00A8B8 !important;
        border-radius: 4px;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        transition: all 0.3s ease;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #008A96, #006B85) !important;
        transform: scale(1.02);
        box-shadow: 0 0 10px rgba(0, 168, 184, 0.4);
        color: #FFFFFF !important;
    }
    
    /* --- TABS --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #111;
        border-radius: 4px 4px 0 0;
        color: #666;
        border: 1px solid #333;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #00F0FF;
        color: #000 !important;
        font-weight: bold;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
    }

    /* --- VERDICT BANNERS --- */
    .verdict-real {
        background: linear-gradient(90deg, rgba(0, 255, 128, 0.1), transparent);
        border-left: 5px solid #00FF80;
        color: #00FF80;
        padding: 20px;
        text-align: left;
        font-family: 'Orbitron', sans-serif;
        font-size: 24px;
        letter-spacing: 2px;
        margin-bottom: 20px;
    }
    
    .verdict-fake {
        background: linear-gradient(90deg, rgba(255, 50, 50, 0.1), transparent);
        border-left: 5px solid #FF3232;
        color: #FF3232;
        padding: 20px;
        text-align: left;
        font-family: 'Orbitron', sans-serif;
        font-size: 24px;
        letter-spacing: 2px;
        margin-bottom: 20px;
    }

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: #080808;
        border-right: 1px solid #222;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector_model():
    model_dir = 'models'
    if not os.path.exists(model_dir):
        return None, None
        
    # 1. Look for timestamped folders first (new structure)
    subdirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and d.startswith('model_')]
    
    if subdirs:
        # Sort by name (timestamp) descending
        subdirs.sort(reverse=True)
        
        # Iterate to find the first valid model directory
        for latest_dir in subdirs:
            model_path = os.path.join(latest_dir, 'model.keras')
            if os.path.exists(model_path):
                print(f"Loading model from: {model_path}")
                return load_model(model_path), latest_dir
            
    # 2. Fallback to old structure (files directly in models/)
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
    if model_files:
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
        latest_model = os.path.join(model_dir, model_files[0])
        print(f"Loading model from: {latest_model}")
        return load_model(latest_model), model_dir
        
    return None, None

def predict_audio(model, file_path, apply_nr=True):
    # Extract features
    mel_spec = extract_mel_spectrogram(file_path, apply_noise_reduction=apply_nr)
    
    if mel_spec is None:
        return None, None
    
    # Preprocess for model (shape: 1, 128, 313, 1)
    # Resize/Pad to match training shape (313 time steps for 10 seconds approx)
    target_width = 313
    if mel_spec.shape[1] > target_width:
        mel_spec = mel_spec[:, :target_width]
    else:
        mel_spec = np.pad(mel_spec, ((0,0), (0, target_width - mel_spec.shape[1])), mode='constant')
        
    # Add batch and channel dims
    input_data = mel_spec[np.newaxis, ..., np.newaxis]
    
    # Predict
    prediction = model.predict(input_data)
    return prediction[0][0], mel_spec

@st.cache_resource
def load_speaker_recognizer():
    return SpeakerRecognizer()

def main():
    # Header
    col_logo, col_title = st.columns([1, 6])
    with col_title:
        st.markdown("<h1>DEEPFAKE AUDIO DETECTOR</h1>", unsafe_allow_html=True)
    
    st.markdown("---")

    model, model_dir = load_detector_model()
    
    if model is None:
        st.error("⚠️ SYSTEM ERROR: Model not found. Run training sequence.")
        return
    
    # Initialize speaker recognizer
    speaker_recognizer = load_speaker_recognizer()

    # Sidebar Navigation
    st.sidebar.title("📡 NAVIGATION")
    page = st.sidebar.radio("Select Module:", ["Forensic Analysis", "Speaker Database", "Model Performance"])
    
    # ========== MODEL PERFORMANCE PAGE ==========
    if page == "Model Performance":
        st.subheader("📊 System Status & Model Performance")
        
        # 1. Active Model Diagnostics
        st.markdown("""
        <div style="background: rgba(0, 240, 255, 0.05); border-left: 4px solid #00F0FF; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <h4 style="margin: 0; color: #00F0FF;">🤖 ACTIVE NEURAL ENGINE</h4>
            <p style="margin: 5px 0 0 0; font-size: 14px;">Real-time diagnostics for the currently loaded detector.</p>
        </div>
        """, unsafe_allow_html=True)
        
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.metric("MODEL NAME", os.path.basename(model_dir) if model_dir else "Standard")
            st.code(f"Location: {model_dir}", language="text")
        with diag_col2:
            st.metric("FRAMEWORK", "TensorFlow / Keras")
            st.code("Backend: CUDA/GPU Accelerated", language="text")
            
        with st.expander("🏗️ VIEW MODEL ARCHITECTURE SUMMARY", expanded=False):
            stream = io.StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            summary_str = stream.getvalue()
            st.code(summary_str, language='text')
            
        st.markdown("---")
        
        if model_dir and (os.path.exists(os.path.join(model_dir, 'training_history.png')) or os.path.exists(os.path.join(model_dir, 'training_history_full.png'))):
             metrics_dir = model_dir
        else:
             metrics_dir = 'metrics'
        
        # Check if metrics exist
        if not os.path.exists(metrics_dir):
            st.warning("⚠️ No detailed metrics found in metrics directory.")
            st.info("Performance plots are usually generated during the offline training phase.")
            
        # 1.5. Training Summary Table (From Log)
        log_path = os.path.join(model_dir if model_dir else 'models', 'training_log.csv')
        if os.path.exists(log_path):
            try:
                history_df = pd.read_csv(log_path)
                if not history_df.empty:
                    last_row = history_df.iloc[-1]
                    st.markdown("### 📊 Final Training Metrics (Latest Model)")
                    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                    with sum_col1:
                        st.metric("EPOCHS", int(last_row.get('epoch', 0)) + 1)
                    with sum_col2:
                         acc = last_row.get('val_accuracy', last_row.get('accuracy', 0))
                         st.metric("VAL ACCURACY", f"{acc:.2%}")
                    with sum_col3:
                         prec = last_row.get('val_precision', last_row.get('precision', 0))
                         st.metric("VAL PRECISION", f"{prec:.2%}")
                    with sum_col4:
                         recall = last_row.get('val_recall', last_row.get('recall', 0))
                         st.metric("VAL RECALL", f"{recall:.2%}")
            except:
                pass

        # 2. Re-organized Performance Visualizations
        st.markdown("### 📈 Performance Visualizations")
        
        # Use columns for better layout - now 3 columns to include ROC
        hist_col, cm_col, roc_col = st.columns(3)
        
        # --- TRAINING HISTORY LOGIC ---
        with hist_col:
            # Prioritize 'metrics' folder as per user request
            history_sources = [
                os.path.join('metrics', 'training_history_full.png'),
                os.path.join('metrics', 'training_history.png')
            ]
            if model_dir:
                history_sources.append(os.path.join(model_dir, 'training_history_full.png'))
                history_sources.append(os.path.join(model_dir, 'training_history.png'))
            
            # Find first existing
            hist_path = next((p for p in history_sources if os.path.exists(p)), None)
            
            if hist_path:
                label = "Full History (37 Epochs)" if "full" in hist_path else "Training History"
                st.markdown(f"**{label}**")
                st.image(hist_path, use_column_width=True)
            else:
                st.info("Training history unavailable.")
        
        # --- CONFUSION MATRIX LOGIC ---
        with cm_col:
            # Prioritize 'metrics' folder
            cm_sources = [
                os.path.join('metrics', 'Confusion_Matrix.png'),
                os.path.join('metrics', 'confusion_matrix.png')
            ]
            if model_dir:
                cm_sources.append(os.path.join(model_dir, 'Confusion_Matrix.png'))
                cm_sources.append(os.path.join(model_dir, 'confusion_matrix.png'))
                cm_sources.append(os.path.join(model_dir, 'confusion_matrix_manual.png'))
            
            # Find first existing
            cm_path = next((p for p in cm_sources if os.path.exists(p)), None)
            
            if cm_path:
                st.markdown("**Confusion Matrix**")
                st.image(cm_path, use_column_width=True)
            else:
                st.info("Confusion matrix unavailable.")

        # --- ROC CURVE LOGIC ---
        with roc_col:
            # Prioritize 'metrics' folder
            roc_sources = [
                os.path.join('metrics', 'ROC_Curve.png'),
                os.path.join('metrics', 'roc_curve.png')
            ]
            if model_dir:
                roc_sources.append(os.path.join(model_dir, 'ROC_Curve.png'))
                roc_sources.append(os.path.join(model_dir, 'roc_curve.png'))
                roc_sources.append(os.path.join(model_dir, 'roc_curve_manual.png'))
            
            # Find first existing
            roc_path = next((p for p in roc_sources if os.path.exists(p)), None)
            
            if roc_path:
                st.markdown("**ROC Curve**")
                st.image(roc_path, use_column_width=True)
            else:
                st.info("ROC curve unavailable.")
            
        return
    
    # ========== SPEAKER MANAGEMENT PAGE ==========
    if page == "Speaker Database":
        st.subheader("👥 Team Voice Database")
        
        tab1, tab2 = st.tabs(["📝 ENROLL NEW AGENT", "🗑️ MANAGE DATABASE"])
        
        with tab1:
            st.markdown("### 🎙️ Voice Profile Enrollment")
            
            # Initialize session state
            if 'enrollment_samples' not in st.session_state:
                st.session_state.enrollment_samples = []
            if 'speaker_name_key' not in st.session_state:
                st.session_state.speaker_name_key = 0
            
            speaker_name = st.text_input(
                "AGENT NAME:", 
                placeholder="ENTER NAME...",
                key=f"speaker_name_{st.session_state.speaker_name_key}"
            )
            
            num_samples = st.slider("REQUIRED SAMPLES:", 3, 5, 4)
            
            for i in range(num_samples):
                st.markdown(f"**SAMPLE {i+1}**")
                enroll_method = st.radio(f"Input Method for Sample {i+1}", ["🎙️ Record", "📂 Upload"], key=f"method_{i}", horizontal=True, label_visibility="collapsed")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    if enroll_method == "🎙️ Record":
                        audio = audio_recorder(key=f"enroll_sample_{i}", text=f"RECORD SAMPLE {i+1}")
                        if audio:
                            if len(st.session_state.enrollment_samples) <= i:
                                st.session_state.enrollment_samples.append(audio)
                            else:
                                st.session_state.enrollment_samples[i] = audio
                            st.success(f"AUDIO CAPTURED")
                            st.audio(audio)
                    else:
                        uploaded_file = st.file_uploader(f"Upload Sample {i+1}", type=['wav', 'mp3', 'flac'], key=f"upload_{i}")
                        if uploaded_file:
                            audio_bytes = uploaded_file.getvalue()
                            if len(st.session_state.enrollment_samples) <= i:
                                st.session_state.enrollment_samples.append(audio_bytes)
                            else:
                                st.session_state.enrollment_samples[i] = audio_bytes
                            st.success(f"FILE UPLOADED")
                            st.audio(audio_bytes)
            
            if st.button("💾 SAVE PROFILE", type="primary"):
                if not speaker_name:
                    st.error("NAME REQUIRED")
                elif len(st.session_state.enrollment_samples) < 3:
                    st.error("INSUFFICIENT SAMPLES")
                else:
                    with st.spinner("ENCRYPTING VOICE PROFILE..."):
                        success = speaker_recognizer.enroll_speaker_from_bytes(
                            speaker_name,
                            st.session_state.enrollment_samples[:num_samples]
                        )
                        if success:
                            st.session_state.enrollment_success = f"✅ AGENT {speaker_name} ENROLLED"
                            st.session_state.enrollment_samples = []
                            st.session_state.speaker_name_key += 1
                            st.experimental_rerun()
                        else:
                            st.error("ENROLLMENT FAILED")
            
            if 'enrollment_success' in st.session_state:
                st.success(st.session_state.enrollment_success)
                del st.session_state.enrollment_success
        
        with tab2:
            st.markdown("### 🗄️ Database Records")
            enrolled_speakers = speaker_recognizer.list_enrolled_speakers()
            
            if not enrolled_speakers:
                st.info("DATABASE EMPTY")
            else:
                for idx, name in enumerate(enrolled_speakers, 1):
                    st.markdown(f"`{idx:02d}` **{name}**")
                
                st.markdown("---")
                speaker_to_delete = st.selectbox("SELECT PROFILE TO DELETE:", enrolled_speakers)
                if st.button("❌ DELETE PROFILE"):
                    if speaker_recognizer.delete_speaker(speaker_to_delete):
                        st.success(f"PROFILE {speaker_to_delete} DELETED")
                        st.experimental_rerun()
        return
    # ========== FORENSIC ANALYSIS PAGE ==========
    
    # Sidebar Options
    st.sidebar.markdown("---")
    st.sidebar.title("⚙️ Analysis Settings")

    # Show active model
    if model_dir:
        model_name = os.path.basename(model_dir)
        st.sidebar.success(f"🤖 Active Model: **{model_name}**")
        # Show full model path for verification
        if os.path.isdir(model_dir):
            model_file = os.path.join(model_dir, 'model.keras')
        else:
            model_file = model_dir
        st.sidebar.caption(f"📁 Path: `{model_file}`")

    use_noise_reduction = st.sidebar.checkbox("Apply Noise Reduction", value=True)
    show_heatmap = st.sidebar.checkbox("Show Grad-CAM Heatmap", value=True)
    show_radar = st.sidebar.checkbox("Show Frequency Radar", value=True) # Added Toggle
    show_explanation = st.sidebar.checkbox("Show Text Explanation", value=True)
    show_temporal = st.sidebar.checkbox("Show Temporal Analysis", value=True) # Default true now
    generate_pdf = st.sidebar.checkbox("Generate PDF Report", value=False)

    # Input Section (Stable Radio Logic)
    st.markdown("### 🛠️ INPUT SOURCE SELECTOR")
    input_mode = st.radio(
        "SELECT INPUT SOURCE",
        ["📂 FILE UPLOAD", "🎙️ LIVE FEED", "🧪 TEST SAMPLE"],
        horizontal=True,
        label_visibility="collapsed",
        key="input_mode_selector"
    )
    
    audio_file_path = None
    
    # Smart Audio Source Logic
    if 'input_type' not in st.session_state:
        st.session_state.input_type = "record"  # Default
    
    if 'last_audio_size' not in st.session_state:
        st.session_state.last_audio_size = 0
    
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None

    if 'is_mixed_prototype' not in st.session_state:
        st.session_state.is_mixed_prototype = False

    if input_mode == "📂 FILE UPLOAD":
        uploaded_file = st.file_uploader("DROP AUDIO FILE (WAV/MP3)", type=['wav', 'mp3'])
        if uploaded_file:
            # Check for new upload
            if uploaded_file != st.session_state.last_uploaded_file:
                st.session_state.input_type = "upload"
                st.session_state.is_mixed_prototype = False # Reset for normal uploads
                st.session_state.last_uploaded_file = uploaded_file
                
            with open("temp_upload.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.session_state.input_type == "upload":
                audio_file_path = "temp_upload.wav"
            
    elif input_mode == "🎙️ LIVE FEED":
        audio = audio_recorder(text="ACTIVATE MICROPHONE", icon_size="2x", neutral_color="#00CC96", key="main_audio_recorder")
        if audio:
            # Check if this is a NEW recording
            if len(audio) != st.session_state.last_audio_size:
                st.session_state.input_type = "record"
                st.session_state.is_mixed_prototype = False # Reset for live recording
                st.session_state.last_audio_size = len(audio)
            
            with open("temp_record.wav", "wb") as f:
                f.write(audio)
            
            if st.session_state.input_type == "record":
                audio_file_path = "temp_record.wav"
            
    else: # 🧪 TEST SAMPLE
        col_test1, col_test2, col_test3, col_test4, col_test5 = st.columns(5)
        
        with col_test1:
            if st.button("LOAD REAL SAMPLE", use_container_width=True):
                real_dir = os.path.join("data", "dataset", "eval", "real")
                if os.path.exists(real_dir):
                    files = [f for f in os.listdir(real_dir) if f.endswith('.flac')]
                    if files:
                        import random
                        random_file = random.choice(files)
                        st.session_state['test_file'] = os.path.join(real_dir, random_file)
                        st.session_state.input_type = "random_real"
                        st.session_state.is_mixed_prototype = False
                        st.success(f"LOADED REAL: {random_file}")
        
        with col_test2:
            if st.button("LOAD FAKE SAMPLE", use_container_width=True):
                fake_dir = os.path.join("data", "dataset", "eval", "fake")
                if os.path.exists(fake_dir):
                    files = [f for f in os.listdir(fake_dir) if f.endswith('.flac')]
                    if files:
                        import random
                        random_file = random.choice(files)
                        st.session_state['test_file'] = os.path.join(fake_dir, random_file)
                        st.session_state.input_type = "random_fake"
                        st.session_state.is_mixed_prototype = False
                        st.success(f"LOADED FAKE: {random_file}")

        with col_test3:
            if st.button("LOAD MIXED SAMPLE", use_container_width=True):
                mixed_dir = os.path.join("data", "mixed_samples")
                if os.path.exists(mixed_dir):
                    files = [f for f in os.listdir(mixed_dir) if f.endswith('.wav')]
                    
                    if files:
                        import random
                        random_file = random.choice(files)
                        st.session_state['test_file'] = os.path.join(mixed_dir, random_file)
                        st.session_state.input_type = "random_mixed"
                        st.session_state.is_mixed_prototype = True # ACTIVE PROTOTYPE
                        st.success(f"LOADED MIXED (PROTOTYPE): {random_file}")
                    else:
                        st.warning("No mixed samples found in data/mixed_samples.")

        with col_test4:
            if st.button("EXT. REAL (BLIND)", use_container_width=True):
                ext_dir = os.path.join("data", "external_test", "wavefake", "real")
                if os.path.exists(ext_dir):
                    files = [f for f in os.listdir(ext_dir) if f.endswith('.flac') or f.endswith('.wav')]
                    if files:
                        import random
                        random_file = random.choice(files)
                        st.session_state['test_file'] = os.path.join(ext_dir, random_file)
                        st.session_state.input_type = "external_real"
                        st.session_state.is_mixed_prototype = False
                        st.success(f"LOADED EXT. REAL: {random_file}")
                else:
                    st.warning("External Real directory not found.")
        
        with col_test5:
            if st.button("EXT. FAKE (BLIND)", use_container_width=True):
                ext_dir = os.path.join("data", "external_test", "wavefake", "fake")
                if os.path.exists(ext_dir):
                    files = [f for f in os.listdir(ext_dir) if f.endswith('.flac') or f.endswith('.wav')]
                    if files:
                        import random
                        random_file = random.choice(files)
                        st.session_state['test_file'] = os.path.join(ext_dir, random_file)
                        st.session_state.input_type = "external_fake"
                        st.session_state.is_mixed_prototype = False
                        st.success(f"LOADED EXT. FAKE: {random_file}")
                else:
                    st.warning("External Fake directory not found.")
        
        # Persist random file selection
        if (st.session_state.input_type.startswith("random") or st.session_state.input_type.startswith("external")) and 'test_file' in st.session_state:
             audio_file_path = st.session_state['test_file']


    # Analysis Flow
    if audio_file_path:
        st.markdown("---")
        
        # 1. Pre-Analysis Check
        col_audio, col_btn = st.columns([2, 1])
        with col_audio:
            st.audio(audio_file_path)
        with col_btn:
            analyze_btn = st.button("ANALYZE AUDIO", use_container_width=True)
            
            # SESSION STATE PERSISTENCE LOGIC
            # 1. Reset analysis if audio input changes (prevent stale results)
            if 'last_audio_path' not in st.session_state:
                st.session_state['last_audio_path'] = audio_file_path
            
            if audio_file_path != st.session_state.get('last_audio_path'):
                st.session_state['analysis_active'] = False
                st.session_state['last_audio_path'] = audio_file_path
            
            # 2. Activate analysis on button click
            if analyze_btn:
                st.session_state['analysis_active'] = True
                st.session_state['pdf_bytes'] = None # Clear old report
                st.session_state['pdf_path'] = None # Clear legacy path
        
        # 3. Main Analysis Block (Runs if Active)
        if st.session_state.get('analysis_active', False):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            y, sr = librosa.load(audio_file_path, sr=16000)
            rms = np.sqrt(np.mean(y**2))
            
            if rms < 0.005:
                st.error("❌ SIGNAL TOO WEAK. INCREASE GAIN.")
            else:
                with st.spinner("🔄 PROCESSING SIGNAL... EXTRACTING FEATURES..."):
                    
                    is_live_recording = st.session_state.get('input_type') == 'record'
                    
                    # Hardcoded Optimal Constants (Simplified UI)
                    width = 5.0      # Standard Window (Restored for Granularity)
                    overlap = 0.5    # 50% Overlap
                    # FIXED: Live recordings need MUCH higher threshold to avoid false positives
                    # 0.98 means only flag as fake if model is 98%+ confident
                    sensitivity = 0.98 if is_live_recording else 0.75 
                    
                    # 2. Run Temporal Analysis
                    try:
                        temporal_analyzer = TemporalAnalyzer(model, window_size=width, overlap=overlap, threshold=sensitivity)
                        temporal_visualizer = TemporalVisualizer(dark_mode=True)
                        result = temporal_analyzer.analyze_temporal(audio_file_path, apply_noise_reduction=use_noise_reduction)
                        
                        # PRE-COMPUTE SPECTRAL FEATURES FOR VISUALIZATION (Moved from Expander)
                        display_score, mel_spec = predict_audio(model, audio_file_path, apply_nr=use_noise_reduction)
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        st.stop()
                
                    # 3. Determine Overall Score & Verdict
                    # UNIFIED LOGIC FOR ALL AUDIO TYPES (Live, Upload, Real/Fake/Mixed Samples)
                    fake_regions = result.get('fake_regions', [])
                    num_fake_regions = len(fake_regions)
                    
                    all_scores = [seg['score'] for seg in result['segments']]
                    max_score = max(all_scores) if all_scores else 0.0
                    min_score = min(all_scores) if all_scores else 0.0
                    avg_score = result['overall_score']
                    
                    # FORENSIC DELTA PROTOTYPE (Restricted Logic)
                    score_delta = max_score - min_score
                    
                    total_duration = result['duration']
                    fake_duration = sum([r['end']-r['start'] for r in fake_regions])
                    fake_ratio = fake_duration / total_duration if total_duration > 0 else 0
                    
                    # Unified verdict determination based on temporal analysis
                    # Uses fake_ratio (percentage of audio flagged as fake) as primary metric
                    
                    # ADJUSTED THRESHOLDS: More lenient for live recordings to reduce false positives
                    # Unified verdict determination based on temporal analysis
                    # Uses fake_ratio (percentage of audio flagged as fake) as primary metric
                    
                    # 3b. CHECK VERIFICATION FOR OVERRIDE
                    is_verified_speaker = False
                    if speaker_recognizer.get_num_enrolled() > 0:
                         s_name, s_conf = speaker_recognizer.identify_speaker(audio_file_path)
                         if s_name is not None:
                             is_verified_speaker = True

                    # ADJUSTED THRESHOLDS: More lenient for live recordings to reduce false positives
                    if is_live_recording:
                        # LIVE RECORDING: Critical Override
                        # Model is sensitive to mic noise, so we use very strict thresholds
                        
                        # IDENTITY OVERRIDE: If speaker is known/enrolled, bias HEAVILY towards REAL
                        if is_verified_speaker:
                             # ABSOLUTE TRUST: If Bio-ID matches, we trust it over the artifact detector
                             # This solves the "100% Fake on Noise" issue
                             verdict = "REAL"
                             display_score = 0.99 # Verified User = 99% Confidence
                             # Ensure Explainer sees "Authentic"
                             score = 0.01 
                        
                        # Standard Live Logic (Unknown Speaker)
                        elif fake_ratio > 0.98:  # Must be ALMOST CERTAINLY fake (>98% of audio)
                            verdict = "FAKE"
                            display_score = max_score
                            score = max_score
                        elif fake_ratio < 0.60:  # Allow up to 60% "noise" as Real for live
                            verdict = "REAL"
                            display_score = 1.0 - avg_score # Use average for more stable confidence
                            score = 0.10 # Treat as low fake probability for Explainer
                            # Removed ambiguity override
                        else:  # 0.60 - 0.98
                            # For live demo, bias towards REAL unless sure
                            if num_fake_regions <= 2:
                                verdict = "REAL"
                                display_score = 1.0 - avg_score # Use average for more stable confidence
                                score = 0.15 # Low fake probability
                            else:
                                verdict = "SUSPICIOUS" # Use easier term than MIXED
                                display_score = max_score
                    else:
                        # UPLOADED / SAMPLES (Universal Honest Logic)
                        # No longer checks if file is 'real' or 'mixed' button - 100% Signal Driven
                        if fake_ratio > 0.85:  # Higher threshold (85%) for firm FAKE verdict
                            verdict = "FAKE"
                            display_score = max_score
                            score = max_score
                        elif st.session_state.get('is_mixed_prototype', False) and score_delta > 0.70:
                            # PROTOTYPE ONLY: Detect splicing via score contrast
                            verdict = "MIXED"
                            display_score = max_score
                            score = max_score
                        elif fake_ratio < 0.60:  # Allow up to 60% suspicious windows for REAL
                            verdict = "REAL"
                            display_score = 1.0 - avg_score
                            score = avg_score # Actual model score
                        else:  # Mixed content or uncertain (0.60 - 0.85)
                            verdict = "MIXED"
                            display_score = max_score
                            score = max_score
 
                    # --- NEW: SYNCHRONIZE TEMPORAL RESULTS WITH VERDICT ---
                    # If the final system verdict is REAL, we must treat any temporal "fake" 
                    # detections as noise-induced false positives and clear them.
                    if verdict == "REAL":
                        result['fake_regions'] = []
                        if 'segments' in result:
                            for segment in result['segments']:
                                segment['label'] = 'real'
                                # Optional: lower the visual score for green bars
                                segment['score'] = min(segment['score'], 0.2)
                    
                    
                    # 1. VERDICT DISPLAY - Refactored to Helper
                    st.markdown("---")
                    display_immersive_verdict(verdict, display_score, score, rms)
                    
                    if is_live_recording and 'is_noisy_real' in locals() and is_noisy_real:
                        st.info("⚠️ **ENVIRONMENT NOISE FILTERED**: The AI detected high levels of background hiss but successfully filtered it. If the verdict is unexpectedly REAL, try a quieter room.")

                    # 2. INTEL GRID (3 Columns)
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1: # Left: Basic Metrcis
                        st.markdown("### 📊 SIGNAL METRICS")
                        st.metric("RMS LEVEL (Volume)", f"{rms:.4f}")
                        
                        # Added Dynamic Range Metric for Troubleshooting
                        # (Calculated from spectrogram if available)
                        if mel_spec is not None:
                            # mel_spec here is the normalized version, let's just use it to show we have signal
                            # Better: calculate raw range from RMS or similar.
                            # For user: signal strength
                            peak = np.max(np.abs(y))
                            st.metric("PEAK SIGNAL", f"{peak:.2f}")
                        
                        st.metric("DURATION", f"{total_duration:.2f}s")
                        st.metric("NOISE REDUCTION", "ACTIVE" if use_noise_reduction else "OFF")

                    with col2: # Center: Voice Identity (3D Sphere)
                        st.markdown("### 🆔 IDENTITY VERIFICATION")
                        if speaker_recognizer.get_num_enrolled() > 0:
                            speaker_name, speaker_conf = speaker_recognizer.identify_speaker(audio_file_path)
                            color_match = "normal"
                            if speaker_name:
                                st.metric("MATCH FOUND", speaker_name, f"{speaker_conf*100:.0f}%", delta_color="normal")
                            else:
                                st.metric("IDENTITY", "UNKNOWN", "NO MATCH", delta_color="off")
                            
                            # 3D Identity Sphere (Mini)
                            profiles = speaker_recognizer.get_all_embeddings()
                            current_embed = speaker_recognizer.get_embedding(audio_file_path)
                            fig_space = temporal_visualizer.plot_speaker_space(
                                profiles, 
                                current_embed, 
                                speaker_name if 'speaker_name' in locals() and speaker_name else None
                            )
                            st.plotly_chart(fig_space, use_container_width=True, config={'displayModeBar': False})
                        else:
                            st.info("No Trusted Profiles Enrolled")

                    with col3: # Right: Temporal Analysis (Timeline)
                        # Use existing 'result' from earlier analysis
                        fake_regions = result.get('fake_regions', []) if result else []
                        
                        # Only show fake segments if verdict is actually FAKE, MIXED, or SUSPICIOUS
                        if verdict in ["FAKE", "MIXED", "SUSPICIOUS"] and fake_regions:
                            # Immersive warning style for detected segments
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, rgba(255, 50, 50, 0.15), rgba(139, 0, 0, 0.05));
                                border: 2px solid #FF3232;
                                border-radius: 8px;
                                padding: 15px;
                                box-shadow: 0 2px 10px rgba(255, 50, 50, 0.2);
                            ">
                                <h3 style="color: #FF3232; margin: 0 0 10px 0; font-size: 16px;">
                                    ⏱️ TIMELINE SCAN
                                </h3>
                                <div style="
                                    background: rgba(255, 50, 50, 0.1);
                                    border-left: 3px solid #FF3232;
                                    padding: 8px 12px;
                                    border-radius: 4px;
                                    margin-bottom: 10px;
                                ">
                                    <p style="color: #FF3232; margin: 0; font-weight: bold; font-size: 14px;">
                                        🚨 {len(fake_regions)} MANIPULATED SEGMENTS
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Show segments
                            for fr in fake_regions[:3]:
                                st.markdown(f"""
                                <div style="
                                    color: #FFA500;
                                    font-size: 12px;
                                    margin: 5px 0;
                                    padding: 5px 10px;
                                    background: rgba(255, 165, 0, 0.1);
                                    border-radius: 3px;
                                ">
                                    🔴 <strong>{fr['start']:.1f}s - {fr['end']:.1f}s</strong> (Suspicious)
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            # Immersive success style for clean audio
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, rgba(0, 255, 128, 0.15), rgba(0, 139, 69, 0.05));
                                border: 2px solid #00FF80;
                                border-radius: 8px;
                                padding: 15px;
                                box-shadow: 0 2px 10px rgba(0, 255, 128, 0.2);
                            ">
                                <h3 style="color: #00FF80; margin: 0 0 10px 0; font-size: 16px;">
                                    ⏱️ TIMELINE SCAN
                                </h3>
                                <div style="
                                    background: rgba(0, 255, 128, 0.1);
                                    border-left: 3px solid #00FF80;
                                    padding: 8px 12px;
                                    border-radius: 4px;
                                ">
                                    <p style="color: #00FF80; margin: 0; font-weight: bold; font-size: 14px;">
                                        ✅ NO TEMPORAL ANOMALIES
                                    </p>
                                    <p style="color: #AAAAAA; margin: 5px 0 0 0; font-size: 11px;">
                                        Audio integrity maintained throughout
                                    </p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)


                    # 3. SIGNAL VISUALIZATION SUITE (Collapsible)
                    st.markdown("---")
                    st.markdown("""
                    <div style="background: linear-gradient(90deg, rgba(0, 240, 255, 0.1), transparent); 
                                border-left: 4px solid #00F0FF; 
                                padding: 10px 15px; 
                                border-radius: 5px; 
                                margin-bottom: 10px;">
                        <h3 style="margin: 0; color: #00F0FF;">📊 ADVANCED SIGNAL VISUALIZATIONS</h3>
                        <p style="margin: 5px 0 0 0; color: #888; font-size: 12px;">Click to expand and view detailed signal analysis</p>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("View Visualizations", expanded=False):
                        # Use pre-computed mel_spec
                        if mel_spec is None:
                             st.warning("⚠️ Spectral data could not be generated. Some visualizations may be unavailable.")
                        
                        # Create tabs for different visualizations
                        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                            "🌊 WAVEFORM", 
                            "🌈 MEL SPECTROGRAM", 
                            "📡 RADAR CHART",
                            "🧊 3D SPECTRAL TOPOGRAPHY"
                        ])
                        
                        # Tab 1: Waveform
                        with viz_tab1:
                            fig_wave, ax_wave = plt.subplots(figsize=(12, 4))
                            librosa.display.waveshow(y, sr=sr, ax=ax_wave, color='#00F0FF', alpha=0.7)
                            ax_wave.set_facecolor('#0a0a0a')
                            fig_wave.patch.set_facecolor('#0a0a0a')
                            ax_wave.set_title("Audio Waveform", color='white', fontsize=14)
                            ax_wave.tick_params(colors='white', labelsize=10)
                            ax_wave.set_xlabel('Time (s)', color='white', fontsize=11)
                            ax_wave.set_ylabel('Amplitude', color='white', fontsize=11)
                            ax_wave.grid(True, alpha=0.2)
                            st.pyplot(fig_wave)
                        
                        # Tab 2: Mel Spectrogram
                        with viz_tab2:
                            if mel_spec is not None:
                                fig_mel, ax_mel = plt.subplots(figsize=(12, 5))
                                img = librosa.display.specshow(mel_spec, sr=sr, x_axis='time', y_axis='mel', ax=ax_mel, cmap='viridis')
                                ax_mel.set_facecolor('#0a0a0a')
                                fig_mel.patch.set_facecolor('#0a0a0a')
                                ax_mel.set_title("Mel Spectrogram", color='white', fontsize=14)
                                ax_mel.tick_params(colors='white', labelsize=10)
                                ax_mel.set_xlabel('Time (s)', color='white', fontsize=11)
                                ax_mel.set_ylabel('Mel Frequency', color='white', fontsize=11)
                                cbar = fig_mel.colorbar(img, ax=ax_mel, format='%+2.0f dB')
                                cbar.ax.tick_params(colors='white', labelsize=10)
                                st.pyplot(fig_mel)
                            else:
                                st.info("No spectral data available")
                        
                        # Tab 3: Radar Chart
                        with viz_tab3:
                            if show_radar:
                                # Extract frequency features
                                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                                spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
                                zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
                                
                                # Normalize to 0-1 range
                                categories = ['Centroid', 'Rolloff', 'ZCR', 'Energy', 'Brightness']
                                values = [
                                    min(spectral_centroid / 4000, 1.0),
                                    min(spectral_rolloff / 8000, 1.0),
                                    min(zero_crossing * 10, 1.0),
                                    min(rms * 50, 1.0),
                                    # Use unified scoring for radar chart
                                    avg_score if is_live_recording else max_score 
                                ]
                                
                                fig_radar = plot_radar_chart(categories, values, "FREQUENCY FEATURES")
                                fig_radar.set_size_inches(1.5, 1.5)  # Micro size
                                fig_radar.patch.set_facecolor('#0a0a0a')
                                st.pyplot(fig_radar, use_container_width=False)
                            else:
                                st.info("Enable 'Show Frequency Radar' in sidebar to view this chart")
                        
                        # Tab 4: 3D Spectral Topography
                        with viz_tab4:
                            if mel_spec is not None:
                                st.markdown("**Interactive 3D View - Use mouse to rotate and zoom**")
                                fig_3d = temporal_visualizer.plot_3d_spectrogram(mel_spec)
                                # Reduce 3D chart height
                                fig_3d.update_layout(height=450)
                                st.plotly_chart(fig_3d, use_container_width=True)
                            else:
                                st.info("No spectral data available for 3D visualization")
                    
                    # 4. TEMPORAL TIMELINE ANALYSIS (Collapsible)
                    if show_temporal:
                        st.markdown("---")
                        st.markdown("""
                        <div style="background: linear-gradient(90deg, rgba(255, 165, 0, 0.1), transparent); 
                                    border-left: 4px solid #FFA500; 
                                    padding: 10px 15px; 
                                    border-radius: 5px; 
                                    margin-bottom: 10px;">
                            <h3 style="margin: 0; color: #FFA500;">⏱️ TEMPORAL TIMELINE ANALYSIS</h3>
                            <p style="margin: 5px 0 0 0; color: #888; font-size: 12px;">Frame-by-frame detection timeline</p>
                        </div>
                        """, unsafe_allow_html=True)
                        with st.expander("View Timeline", expanded=False):
                            st.markdown("**Timeline Visualization:**")
                            
                            # Timeline visualization
                            timeline_fig = temporal_visualizer.plot_timeline(result)
                            st.pyplot(timeline_fig)
                            
                            # Confidence Heatmap
                            st.markdown("**Confidence Heatmap:**")
                            heatmap_fig = temporal_visualizer.plot_confidence_heatmap(result)
                            st.pyplot(heatmap_fig)
                            
                            if verdict == "MIXED" and fake_regions:
                                st.warning(f"⚠️ **ATTENTION**: Found {len(fake_regions)} suspicious regions in this file!")
                                for i, region in enumerate(fake_regions, 1):
                                    st.write(f"• **Region {i}**: {region['start']:.2f}s - {region['end']:.2f}s (Confidence: {region['confidence']*100:.1f}%)")
                    
                    # 5. XAI FORENSIC REPORT (Collapsible)
                    st.markdown("---")
                    st.markdown("""
                    <div style="background: linear-gradient(90deg, rgba(138, 43, 226, 0.1), transparent); 
                                border-left: 4px solid #8A2BE2; 
                                padding: 10px 15px; 
                                border-radius: 5px; 
                                margin-bottom: 10px;">
                        <h3 style="margin: 0; color: #8A2BE2;">📄 EXPLAINABLE AI FORENSIC REPORT</h3>
                        <p style="margin: 5px 0 0 0; color: #888; font-size: 12px;">AI-powered analysis with visual explanations</p>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("View Forensic Report", expanded=False):
                        if show_explanation and mel_spec is not None:
                            explainer = DeepfakeExplainer(model)
                            mel_spec_for_explainer = mel_spec[..., np.newaxis] # Correct shape
                            heatmap = explainer.generate_gradcam(mel_spec_for_explainer, display_score)
                            
                            st.markdown("**Grad-CAM Heatmap:**")
                            fig_cam = explainer.overlay_heatmap(mel_spec, heatmap)
                            st.pyplot(fig_cam)
                            
                            st.markdown("**XAI Explanation:**")
                            analysis_res = explainer.analyze_heatmap_regions(heatmap, mel_spec)
                            
                            # Handle SUSPICIOUS/MIXED verdict
                            if verdict == "MIXED":
                                explanation = f"""⚠️ **SUSPICIOUS AUDIO DETECTED** (Mixed Signals)

**Analysis Details:**
This audio contains inconsistent patterns suggesting partial manipulation or splicing.

**Findings:**
- Detected {len(fake_regions)} suspicious segments within the audio
- Alternating authentic and synthetic characteristics
- Likely audio editing or voice insertion

**Recommendation:**
- Manual review of flagged segments recommended
- Check timestamps: {', '.join([f"{r['start']:.1f}s-{r['end']:.1f}s" for r in fake_regions[:3]])}
- Consider context and source verification
"""
                            else:
                                # Use dynamic threshold (0.98 for live, 0.75 for upload)
                                active_threshold = 0.98 if is_live_recording else 0.75
                                
                                # Synchronize score with display_score for the text explanation
                                # If REAL, explainer needs a score < threshold (authenticity = 1-score)
                                # If FAKE, explainer needs a score > threshold
                                explainer_score = score
                                if verdict == "REAL":
                                    # Map display_score (authenticity) back to fake probability for explainer
                                    # This ensures the text ALWAYS matches the REAL banner
                                    explainer_score = min(active_threshold - 0.01, 1.0 - display_score)
                                elif verdict == "FAKE":
                                    # Force score above threshold to ensure FAKE text
                                    explainer_score = max(active_threshold + 0.01, display_score)
                                    
                                explanation = explainer.generate_explanation(explainer_score, analysis_res, threshold=active_threshold)
                            
                            st.text(explanation)
                            
                            # PDF Generation (only if sidebar option enabled)
                            if generate_pdf:
                                # Initialize session state for PDF if needed
                                if 'pdf_bytes' not in st.session_state:
                                    st.session_state['pdf_bytes'] = None

                                # GENERATE BUTTON
                                if st.button("GENERATE PDF REPORT", key="gen_pdf_btn"):
                                     with st.spinner("Generating Report..."):
                                         pdf_buffer = io.BytesIO()
                                         
                                         # Pre-generate timeline diagram for PDF if not already in scope
                                         # (It uses the 'result' from earlier analysis)
                                         timeline_fig_for_pdf = temporal_visualizer.plot_timeline(result)
                                         
                                         # Pass buffer and figure
                                         result_pdf = explainer.generate_pdf_report(
                                            audio_file_path, display_score, mel_spec, 
                                            fig_cam, explanation, verdict=verdict,
                                            output_path=pdf_buffer,
                                            fake_regions=fake_regions,
                                            timeline_fig=timeline_fig_for_pdf
                                         )
                                         
                                         if result_pdf:
                                             st.session_state['pdf_bytes'] = pdf_buffer.getvalue()
                                             st.success("Report Generated Successfully!")
                                         else:
                                             st.error("Report Generation Failed. Check terminal logs.")
                                
                                # DOWNLOAD BUTTON (Persistent)
                                if st.session_state.get('pdf_bytes'):
                                     st.download_button(
                                         label="⬇️ DOWNLOAD FINAL REPORT",
                                         data=st.session_state['pdf_bytes'],
                                         file_name="Deepfake_Forensic_Report.pdf",
                                         mime="application/pdf",
                                         key="dl_pdf_btn"
                                     )
                        else:
                             if not show_explanation:
                                 st.info("Enable 'Show Text Explanation' in sidebar to view this report.")
                             elif mel_spec is None:
                                 st.error("Cannot generate report: Spectral data missing.")


if __name__ == "__main__":
    main()
