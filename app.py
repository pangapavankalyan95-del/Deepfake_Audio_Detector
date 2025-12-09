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
    
    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#FF3232')
    ax.fill(angles, values, '#FF3232', alpha=0.25)
    
    plt.xticks(angles[:-1], categories, size=8, color='#8B949E')
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=6)
    plt.ylim(0, 1)
    plt.title(title, size=10, y=1.1, color='white')
    return fig

# --- Preprocessing Functions ---
def load_audio(file_path, sr=16000, duration=5):
    """
    Loads an audio file and pads/truncates it to a fixed duration.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        # Trim silence/clicks from start and end
        y, _ = librosa.effects.trim(y, top_db=20)
        
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

def extract_mel_spectrogram(file_path, sr=16000, n_mels=128, duration=5, apply_noise_reduction=True):
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
    
    # Normalize to [0, 1] (Standard Min-Max matching training data)
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_db

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
        latest_dir = subdirs[0]
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
    
    # Preprocess for model (shape: 1, 128, 157, 1)
    # Resize/Pad to match training shape (157 time steps for 5 seconds)
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
    speaker_recognizer = SpeakerRecognizer()

    # Sidebar Navigation
    st.sidebar.title("📡 NAVIGATION")
    page = st.sidebar.radio("Select Module:", ["Forensic Analysis", "Speaker Database", "Model Performance"])
    
    # ========== MODEL PERFORMANCE PAGE ==========
    if page == "Model Performance":
        st.subheader("📊 Model Performance Metrics")
        
        if model_dir and os.path.exists(os.path.join(model_dir, 'training_history.png')):
             metrics_dir = model_dir
        else:
             metrics_dir = 'metrics'
        
        # Check if metrics exist
        if not os.path.exists(metrics_dir):
            st.warning("⚠️ No metrics found. Please train the model first using the Jupyter notebook.")
            st.info("Run `Deepfake_Detection_Complete.ipynb` to generate performance metrics.")
            return
        
        # Display accuracy and loss plots
        # Check for new name (training_history.png) or old name (accuracy_loss.png)
        hist_path = os.path.join(metrics_dir, 'training_history.png')
        if not os.path.exists(hist_path):
            hist_path = os.path.join(metrics_dir, 'accuracy_loss.png')

        if os.path.exists(hist_path):
            st.markdown("### 📈 Training History")
            st.image(hist_path, use_column_width=True)
        else:
            st.info("Training history plot not found.")
        
        st.markdown("---")
        
        # Display confusion matrix
        cm_path = os.path.join(metrics_dir, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            st.markdown("### 🎯 Confusion Matrix")
            st.image(cm_path, use_column_width=True)
        else:
            st.info("Confusion matrix not found.")
        
        st.markdown("---")
        
        # Display classification report
        # Check for image first, then text
        cr_img_path = os.path.join(metrics_dir, 'classification_report.png')
        cr_txt_path = os.path.join(metrics_dir, 'classification_report.txt')
        
        st.markdown("### 📋 Classification Report")
        if os.path.exists(cr_img_path):
            st.image(cr_img_path, use_column_width=True)
        elif os.path.exists(cr_txt_path):
            with open(cr_txt_path, 'r') as f:
                report_text = f.read()
            st.code(report_text, language='text')
        else:
            st.info("Classification report not found.")
        
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
                speaker_to_delete = st.selectbox("SELECT PROFILE TO PURGE:", enrolled_speakers)
                if st.button("❌ PURGE PROFILE"):
                    if speaker_recognizer.delete_speaker(speaker_to_delete):
                        st.success(f"PROFILE {speaker_to_delete} PURGED")
                        st.experimental_rerun()
        return
    # ========== FORENSIC ANALYSIS PAGE ==========
    
    # Sidebar Options
    st.sidebar.markdown("---")
    st.sidebar.title("⚙️ Analysis Settings")

    # Show active model
    if model_dir:
        model_name = os.path.basename(model_dir)
        st.sidebar.success(f"🤖 Model Active: {model_name}")

    use_noise_reduction = st.sidebar.checkbox("Apply Noise Reduction", value=True)
    show_heatmap = st.sidebar.checkbox("Show Grad-CAM Heatmap", value=True)
    show_radar = st.sidebar.checkbox("Show Frequency Radar", value=True) # Added Toggle
    show_explanation = st.sidebar.checkbox("Show Text Explanation", value=True)
    show_temporal = st.sidebar.checkbox("Show Temporal Analysis", value=True) # Default true now
    generate_pdf = st.sidebar.checkbox("Generate PDF Report", value=False)

    # Input Section (Tabs)
    input_tab1, input_tab2, input_tab3 = st.tabs(["📂 FILE UPLOAD", "🎙️ LIVE FEED", "🧪 TEST SAMPLE"])
    
    audio_file_path = None
    
    # Smart Audio Source Logic
    if 'input_type' not in st.session_state:
        st.session_state.input_type = "record"  # Default
    
    if 'last_audio_size' not in st.session_state:
        st.session_state.last_audio_size = 0
    
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None

    with input_tab1:
        uploaded_file = st.file_uploader("DROP AUDIO FILE (WAV/MP3)", type=['wav', 'mp3'])
        if uploaded_file:
            # Check for new upload
            if uploaded_file != st.session_state.last_uploaded_file:
                st.session_state.input_type = "upload"
                st.session_state.last_uploaded_file = uploaded_file
                
            with open("temp_upload.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.session_state.input_type == "upload":
                audio_file_path = "temp_upload.wav"
            
    with input_tab2:
        audio = audio_recorder(text="ACTIVATE MICROPHONE", icon_size="2x", neutral_color="#00CC96")
        if audio:
            # Check if this is a NEW recording
            if len(audio) != st.session_state.last_audio_size:
                st.session_state.input_type = "record"
                st.session_state.last_audio_size = len(audio)
            
            with open("temp_record.wav", "wb") as f:
                f.write(audio)
            
            if st.session_state.input_type == "record":
                audio_file_path = "temp_record.wav"
            
    with input_tab3:
        col_test1, col_test2, col_test3 = st.columns(3)
        
        with col_test1:
            if st.button("LOAD REAL SAMPLE", use_container_width=True):
                real_dir = os.path.join("data", "dataset", "train", "real")
                if os.path.exists(real_dir):
                    files = [f for f in os.listdir(real_dir) if f.endswith('.flac')]
                    if files:
                        import random
                        random_file = random.choice(files)
                        st.session_state['test_file'] = os.path.join(real_dir, random_file)
                        st.session_state.input_type = "random_real"
                        st.success(f"LOADED REAL: {random_file}")
        
        with col_test2:
            if st.button("LOAD FAKE SAMPLE", use_container_width=True):
                fake_dir = os.path.join("data", "dataset", "train", "fake")
                if os.path.exists(fake_dir):
                    files = [f for f in os.listdir(fake_dir) if f.endswith('.flac')]
                    if files:
                        import random
                        random_file = random.choice(files)
                        st.session_state['test_file'] = os.path.join(fake_dir, random_file)
                        st.session_state.input_type = "random_fake"
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
                        st.success(f"LOADED MIXED: {random_file}")
                    else:
                        st.warning("No mixed samples found in data/mixed_samples.")
        
        # Persist random file selection
        if st.session_state.input_type.startswith("random") and 'test_file' in st.session_state:
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
            
        if analyze_btn:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            y, sr = librosa.load(audio_file_path, sr=16000)
            rms = np.sqrt(np.mean(y**2))
            
            if rms < 0.005:
                st.error("❌ SIGNAL TOO WEAK. INCREASE GAIN.")
            else:
                with st.spinner("🔄 PROCESSING SIGNAL... EXTRACTING FEATURES..."):
                    
                    # Hardcoded Optimal Constants (Simplified UI)
                    width = 5.0      # Standard Window
                    overlap = 0.5    # 50% Overlap
                    sensitivity = 0.10 # Optimal Sensitivity for Forensics
                    
                    # 2. Run Temporal Analysis
                    try:
                        temporal_analyzer = TemporalAnalyzer(model, window_size=width, overlap=overlap, threshold=sensitivity)
                        temporal_visualizer = TemporalVisualizer(dark_mode=True)
                        result = temporal_analyzer.analyze_temporal(audio_file_path, apply_noise_reduction=use_noise_reduction)
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        st.stop()
                
                if result:
                    # 3. Determine Overall Score & Verdict
                    fake_regions = result.get('fake_regions', [])
                    num_fake_regions = len(fake_regions)
                    
                    all_scores = [seg['score'] for seg in result['segments']]
                    max_score = max(all_scores) if all_scores else 0.0
                    avg_score = result['overall_score']
                    
                    total_duration = result['duration']
                    fake_duration = sum([r['end']-r['start'] for r in fake_regions])
                    fake_ratio = fake_duration / total_duration if total_duration > 0 else 0
                    
                    if fake_ratio > 0.95:
                        verdict = "FAKE"
                        display_score = max_score
                    elif fake_ratio < 0.05 and num_fake_regions == 0:
                        verdict = "REAL"
                        display_score = 1.0 - max_score
                    else:
                        # MIXED / SUSPICIOUS CASE
                        # User Rule: If Live Recording, strict Real/Fake (Noise is noisy).
                        # If Upload, show Suspicious + Segments.
                        is_live_recording = st.session_state.get('input_type') == 'record'
                        
                        if is_live_recording:
                            # Force Binary Verdict for Live Mic
                            if max_score > 0.5: 
                                verdict = "FAKE"
                                display_score = max_score
                            else:
                                verdict = "REAL"
                                display_score = 1.0 - max_score
                        else:
                            verdict = "MIXED"
                            display_score = max_score

                    # Define 'score' for downstream components (XAI, Radar)
                    score = max_score 
                    
                    # --- RESULTS DASHBOARD ---
                    
                    # 1. Verdict Banner
                    if verdict == "FAKE":
                        st.markdown(f"""
                        <div class="verdict-fake">
                            ⚠️ DEEPFAKE DETECTED<br>
                            <span style="font-size:16px; color:#FF3232">CONFIDENCE: {display_score*100:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                    elif verdict == "MIXED":
                         st.markdown(f"""
                        <div class="verdict-fake" style="border-color: #FFA500; color: #FFA500; background: linear-gradient(90deg, rgba(255, 165, 0, 0.1), transparent);">
                            ⚠️ SUSPICIOUS / MIXED AUDIO<br>
                            <span style="font-size:16px; color:#FFA500">POTENTIAL MANIPULATION DETECTED ({num_fake_regions} Regions)</span>
                        </div>
                        """, unsafe_allow_html=True)
                         
                         # SEGMENT BREAKDOWN (Requested Feature)
                         # "Real from x to y, Fake from z to w"
                         st.markdown("### 🕵️ DETAILED SEGMENT BREAKDOWN")
                         seg_col1, seg_col2 = st.columns(2)
                         with seg_col1:
                             st.markdown("**🔴 FAKE SEGMENTS:**")
                             if fake_regions:
                                 for fr in fake_regions:
                                     st.markdown(f"- **{fr['start']:.1f}s - {fr['end']:.1f}s** (Conf: {fr['confidence']*100:.0f}%)")
                             else:
                                 st.markdown("- None strongly detected")
                         
                         with seg_col2:
                             st.markdown("**🟢 REAL SEGMENTS:**")
                             # Calculate Real segments by inverting fake ones
                             # This is a simple approximation for the UI
                             cursor = 0.0
                             real_segs = []
                             sorted_fakes = sorted(fake_regions, key=lambda x: x['start'])
                             for fr in sorted_fakes:
                                 if fr['start'] > cursor + 0.5: # 0.5s tolerance
                                     real_segs.append((cursor, fr['start']))
                                 cursor = max(cursor, fr['end'])
                             if total_duration > cursor + 0.5:
                                 real_segs.append((cursor, total_duration))
                                 
                             if real_segs:
                                 for start, end in real_segs:
                                     st.markdown(f"- **{start:.1f}s - {end:.1f}s**")
                             else:
                                 st.markdown("- No clear real segments")
                         st.markdown("---")

                    else: # REAL
                        st.markdown(f"""
                        <div class="verdict-real">
                            ✅ REAL AUDIO<br>
                            <span style="font-size:16px; color:#00FF80">CONFIDENCE: {display_score*100:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # 2. Key Metrics (Always Visible)
                    st.markdown("#### 📊 KEY METRICS")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if verdict == "MIXED":
                             st.metric("VERDICT", "SUSPICIOUS", f"{time_str(fake_duration)} FAKE", delta_color="off")
                        else:
                             st.metric("CONFIDENCE SCORE", f"{display_score*100:.1f}%", delta=verdict, delta_color="inverse" if verdict=="FAKE" else "normal")
                    
                    with col2:
                        st.metric("RMS LEVEL (Volume)", f"{rms:.4f}")
                    
                    with col3:
                        # Speaker ID
                        if speaker_recognizer.get_num_enrolled() > 0:
                            speaker_name, speaker_conf = speaker_recognizer.identify_speaker(audio_file_path)
                            if speaker_name:
                                st.metric("VOICE ID MATCH", speaker_name, f"{speaker_conf*100:.1f}%")
                            else:
                                st.metric("VOICE ID", "UNKNOWN", "NO MATCH")
                        else:
                            st.info("SPEAKER DB EMPTY")
                            
                    # 3. SIGNAL DETAILS (Standard Analysis - Visible)
                    st.markdown("---")
                    st.markdown("### 🎵 SIGNAL ANALYSIS (Standard)")
                    
                    viz_col1, viz_col2 = st.columns([2, 1])
                    
                    # Generate mel_spec for visualization
                    try:
                         _, mel_spec = predict_audio(model, audio_file_path, apply_nr=use_noise_reduction)
                    except:
                         mel_spec = None

                    with viz_col1:
                        viz_tab1, viz_tab2 = st.tabs(["🌊 WAVEFORM", "🌈 MEL SPECTROGRAM"])
                        with viz_tab1:
                            fig_wave, ax_wave = plt.subplots(figsize=(8, 3))
                            librosa.display.waveshow(y, sr=sr, ax=ax_wave, color='blue', alpha=0.6)
                            ax_wave.set_facecolor('#0a0a0a')
                            fig_wave.patch.set_facecolor('#0a0a0a')
                            ax_wave.set_title("Audio Waveform", color='white', fontsize=10)
                            ax_wave.tick_params(colors='white', labelsize=8)
                            ax_wave.set_xlabel('Time (s)', color='white', fontsize=8)
                            st.pyplot(fig_wave)
                            
                        with viz_tab2:
                            if mel_spec is not None:
                                fig_mel, ax_mel = plt.subplots(figsize=(8, 3))
                                img = librosa.display.specshow(mel_spec, sr=sr, x_axis='time', y_axis='mel', ax=ax_mel, cmap='viridis')
                                ax_mel.set_facecolor('#0a0a0a')
                                fig_mel.patch.set_facecolor('#0a0a0a')
                                ax_mel.set_title("Mel Spectrogram", color='white', fontsize=10)
                                ax_mel.tick_params(colors='white', labelsize=8)
                                ax_mel.set_xlabel('Time (s)', color='white', fontsize=8)
                                cbar = fig_mel.colorbar(img, ax=ax_mel, format='%+2.0f dB')
                                cbar.ax.tick_params(colors='white', labelsize=8)
                                st.pyplot(fig_mel)
                    
                    with viz_col2:
                         if show_radar:
                            # Frequency Analysis Radar Chart
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
                                score 
                            ]
                            
                            fig_radar = plot_radar_chart(categories, values, "FREQUENCY FEATURES")
                            fig_radar.patch.set_facecolor('#0a0a0a')
                            st.pyplot(fig_radar)
                            
                    # 4. TIMELINE VISUALIZATION (Moved Down)
                    if show_temporal:
                        st.markdown("---")
                        st.markdown("### ⏱️ TEMPORAL TIMELINE (Whole File)")
                        
                        # Timeline visualization
                        timeline_fig = temporal_visualizer.plot_timeline(result)
                        st.pyplot(timeline_fig)
                        
                        # Confidence Heatmap
                        heatmap_fig = temporal_visualizer.plot_confidence_heatmap(result)
                        st.pyplot(heatmap_fig)
                        
                        if verdict == "MIXED" and fake_regions:
                             st.warning(f"⚠️ **ATTENTION**: Found {len(fake_regions)} suspicious regions in this file!")
                             for i, region in enumerate(fake_regions, 1):
                                st.write(f"• **Region {i}**: {region['start']:.2f}s - {region['end']:.2f}s (Confidence: {region['confidence']*100:.1f}%)")
                    
                    # 5. XAI Analysis (Collapsible)
                    if show_heatmap or show_explanation:
                        with st.expander("🧠 EXPLAINABLE AI ANALYSIS", expanded=False):
                            st.markdown("---")
                            st.subheader("🧠 DETAILED ANALYSIS")
                            
                            explainer = DeepfakeExplainer(model)
                            
                            # Prepare mel_spec with correct shape for explainer (128, 157, 1)
                            if mel_spec is not None:
                                mel_spec_for_explainer = mel_spec[..., np.newaxis]
                                
                                if show_heatmap:
                                    st.markdown("**GRAD-CAM HEATMAP (Suspicious Regions):**")
                                    with st.spinner("GENERATING HEATMAP..."):
                                        try:
                                            # Generate heatmap
                                            heatmap = explainer.generate_gradcam(mel_spec_for_explainer, score)
                                            
                                            # Overlay on mel spectrogram
                                            heatmap_fig = explainer.overlay_heatmap(mel_spec, heatmap)
                                            st.pyplot(heatmap_fig)
                                        except Exception as e:
                                            st.error(f"Error generating Grad-CAM: {e}")
                                
                                if show_explanation:
                                    with st.spinner("ANALYZING PATTERNS..."):
                                        try:
                                            # Analyze heatmap regions if heatmap was generated
                                            if show_heatmap and 'heatmap' in locals():
                                                analysis_results = explainer.analyze_heatmap_regions(heatmap, mel_spec)
                                            else:
                                                # Generate heatmap just for analysis
                                                heatmap = explainer.generate_gradcam(mel_spec_for_explainer, score)
                                                analysis_results = explainer.analyze_heatmap_regions(heatmap, mel_spec)
                                            
                                            # Generate explanation text
                                            if verdict == "MIXED":
                                                 explanation = f"""
### ⚠️ SUSPICIOUS / MIXED AUDIO REPORT

**Analysis:**
This audio file contains inconsistent features. The forensic model detected specific regions of manipulation within an otherwise authentic-sounding file.

**Potential Manipulation Type:**
- **Audio Splicing / Insertion:** The alternation between Real and Fake segments suggests that words or phrases may have been inserted or altered (e.g., changing "like" to "did not like").

**Temporal Inconsistency:**
- **Fake Regions:** {num_fake_regions} detected segments (See breakdown above).
- **Inconsistency:** The presence of both high-confidence real and fake segments is a strong indicator of tampering.

**Recommendation:**
Manual review is recommended for the specific timestamps flagged in the breakdown above.
"""
                                            else:
                                                 explanation = explainer.generate_explanation(score, analysis_results)
                                            
                                            st.markdown("### 📄 READ ANALYSIS REPORT")
                                            st.markdown(explanation)
                                        except Exception as e:
                                            st.error(f"Error generating explanation: {e}")
                                            explanation = ""
                    
                    # 6. PDF Report Generation (Conditional - Only if enabled in sidebar)
                    if generate_pdf:
                        st.markdown("---")
                        with st.spinner("GENERATING PDF REPORT..."):
                            try:
                                # Save plots to bytes for report
                                buf_mel = io.BytesIO()
                                if 'fig_mel' in locals():
                                    fig_mel.savefig(buf_mel, format='png', bbox_inches='tight', facecolor='#0a0a0a')
                                buf_mel.seek(0)
                                
                                buf_radar = io.BytesIO()
                                if 'fig_radar' in locals():
                                    fig_radar.savefig(buf_radar, format='png', bbox_inches='tight', facecolor='#0a0a0a')
                                buf_radar.seek(0)
                                
                                plot_images_dict = {
                                    'mel_spec': buf_mel,
                                    'radar': buf_radar
                                }
                                
                                # Add temporal timeline if available
                                if show_temporal and 'result' in locals() and result is not None:
                                    buf_timeline = io.BytesIO()
                                    timeline_fig.savefig(buf_timeline, format='png', bbox_inches='tight', facecolor='#0a0a0a')
                                    buf_timeline.seek(0)
                                    plot_images_dict['timeline'] = buf_timeline
                                
                                # Generate Report
                                report_gen = ForensicReportGenerator()
                                pdf_bytes = report_gen.generate_report(
                                    audio_filename=uploaded_file.name if uploaded_file else "Live_Recording.wav",
                                    prediction_score=score,
                                    threshold=0.5,
                                    speaker_id=speaker_name if 'speaker_name' in locals() and speaker_name else "Unknown",
                                    speaker_conf=speaker_conf if 'speaker_conf' in locals() else 0.0,
                                    explanation=explanation if 'explanation' in locals() else "No explanation available.",
                                    plot_images=plot_images_dict,
                                    temporal_result=result if show_temporal and 'result' in locals() else None,
                                    verdict=verdict # Pass verdict for styling
                                )
                                
                                st.download_button(
                                    label="📄 DOWNLOAD FORENSIC REPORT (PDF)",
                                    data=pdf_bytes,
                                    file_name=f"Forensic_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf"
                                )
                            except Exception as e:
                                st.error(f"Error generating PDF report: {e}")

if __name__ == "__main__":
    main()
