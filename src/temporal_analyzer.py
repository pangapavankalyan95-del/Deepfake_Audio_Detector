"""
Temporal Deepfake Detection Module
Analyzes audio to identify WHEN manipulation occurred (not just IF it's fake).

Uses sliding window approach with existing deepfake detection model.
"""

import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import noisereduce as nr

class TemporalAnalyzer:
    """
    Analyzes audio temporally to detect when deepfake manipulation occurred.
    """
    
    def __init__(self, model, window_size=5.0, overlap=0.5, threshold=0.75):
        """
        Initialize temporal analyzer.
        
        Args:
            model: Trained deepfake detection model
            window_size: Size of analysis window in seconds (default: 5.0)
            overlap: Overlap ratio between windows (default: 0.5 = 50%)
            threshold: Classification threshold (default: 0.3, lowered for mixed audio)
        """
        self.model = model
        self.window_size = window_size
        self.overlap = overlap
        self.threshold = threshold
        self.target_sr = 16000
        
    def analyze_temporal(self, audio_path: str, apply_noise_reduction: bool = True) -> Dict:
        """
        Perform temporal analysis on audio file.
        
        Args:
            audio_path: Path to audio file
            apply_noise_reduction: Whether to apply noise reduction
            
        Returns:
            Dictionary containing:
                - segments: List of time segments with predictions
                - fake_regions: List of detected fake regions
                - overall_score: Overall fake probability
                - timeline: Frame-by-frame predictions
        """
        # Load audio
        audio, _ = librosa.load(audio_path, sr=self.target_sr)
        
        # Trimming & Normalization removed to match model training distribution
        # audio, _ = librosa.effects.trim(audio, top_db=40)
        # audio = librosa.util.normalize(audio)
        
        duration = len(audio) / self.target_sr
        
        # Calculate window parameters
        window_samples = int(self.window_size * self.target_sr)
        hop_samples = int(window_samples * (1 - self.overlap))
        
        # Analyze each window
        segments = []
        predictions = []
        
        start_sample = 0
        while start_sample < len(audio):
            end_sample = min(start_sample + window_samples, len(audio))
            
            # Extract window
            window_audio = audio[start_sample:end_sample]
            
            # Removed tiling logic. Raw signal is better.
            # Spectrogram will be padded to 313 frames in _get_single_prediction.
            
            # Get prediction for this window
            try:
                score = self._predict_window(window_audio, self.target_sr, apply_noise_reduction)
            except Exception as e:
                print(f"Error predicting window: {e}")
                score = 0.0
            
            # Calculate timestamps
            start_time = start_sample / self.target_sr
            end_time = end_sample / self.target_sr
            
            segments.append({
                'start': round(start_time, 2),
                'end': round(end_time, 2),
                'score': round(float(score), 4),
                'label': 'fake' if score > self.threshold else 'real'
            })
            
            predictions.append(score)
            
            # Move to next window
            start_sample += hop_samples
        
        # Smooth predictions
        smoothed_predictions = self._smooth_predictions(predictions)
        
        # Extract fake regions
        fake_regions = self._extract_fake_regions(segments, smoothed_predictions)
        
        # Calculate overall score
        overall_score = np.mean(predictions) if predictions else 0.0
        
        return {
            'duration': round(duration, 2),
            'segments': segments,
            'fake_regions': fake_regions,
            'overall_score': round(float(overall_score), 4),
            'num_segments': len(segments),
            'window_size': self.window_size,
            'overlap': self.overlap
        }
    
    def _predict_window(self, audio: np.ndarray, sr: int, apply_nr: bool) -> float:
        """
        Predict fake probability for a single audio window.
        Uses recursive sub-window scanning if sensitivity is high (threshold < 0.25).
        """
        try:
            # SILENCE GUARD: If window is mostly silent, it's NOT a deepfake (it's just noise)
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.002 or len(audio) < 2048: # Silence or too short for FFT
                return 0.0
                
            # 1. Main prediction on full window
            main_score = self._get_single_prediction(audio, sr, apply_nr)
            
            # High Sensitivity Check (Sub-window scanning)
            # REMOVED: Tiling creates artifacts that trigger false positives in real audio.
            # Single pass on 5s window is more robust than tiling 0.5s chunks.
            
            return float(main_score)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0

    def _get_single_prediction(self, audio: np.ndarray, sr: int, apply_nr: bool, nr_strength: float = 0.8) -> float:
        """Internal helper for raw model prediction."""
        # Extract features directly from audio array
        mel_spec = self._extract_features_from_audio(audio, sr, apply_nr, nr_strength)
        
        if mel_spec is None:
            return 0.0
        
        # Resize/pad to match training shape (313 time steps for 10s audio)
        target_width = 313
        current_width = mel_spec.shape[1]
        
        if current_width < target_width:
            pad_width = target_width - current_width
            mel_spec = np.pad(mel_spec, ((0,0), (0, pad_width)), mode='constant')
        elif current_width > target_width:
            mel_spec = mel_spec[:, :target_width]
        
        # Add batch and channel dims
        input_data = mel_spec[np.newaxis, ..., np.newaxis]
        
        # Predict
        prediction = self.model.predict(input_data, verbose=0)
        score = prediction[0][0]  # Model outputs: 0=Real, 1=Fake
        return score

    def _extract_features_from_audio(self, y: np.ndarray, sr: int, apply_nr: bool, nr_strength: float = 0.8) -> Optional[np.ndarray]:
        """
        Extract Mel Spectrogram from audio numpy array.
        """
        try:
            if apply_nr:
                try:
                    y = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=nr_strength)
                except Exception:
                    pass # Skip if NR fails
            
            # Compute Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Robust Normalization: Use 60dB floor to prevent noise magnification
            # Matches app.py logic to ensure windows without clear speech don't blow up noise
            db_range = mel_spec_db.max() - mel_spec_db.min()
            norm_range = max(db_range, 60.0) # 60dB Minimum dynamic range
            
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (norm_range + 1e-8)
            
            return mel_spec_norm
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def _smooth_predictions(self, predictions: List[float], window_size: int = 3) -> List[float]:
        """
        Smooth predictions using moving average.
        
        Args:
            predictions: List of prediction scores
            window_size: Size of smoothing window
            
        Returns:
            Smoothed predictions
        """
        if len(predictions) < window_size:
            return predictions
        
        smoothed = []
        for i in range(len(predictions)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(predictions), i + window_size // 2 + 1)
            window = predictions[start_idx:end_idx]
            smoothed.append(np.mean(window))
        
        return smoothed
    
    def _extract_fake_regions(self, segments: List[Dict], smoothed_predictions: List[float]) -> List[Dict]:
        """
        Extract continuous fake regions from segments.
        
        Args:
            segments: List of segment dictionaries
            smoothed_predictions: Smoothed prediction scores
            
        Returns:
            List of fake regions with start/end times and confidence
        """
        fake_regions = []
        current_region = None
        
        for i, (segment, score) in enumerate(zip(segments, smoothed_predictions)):
            is_fake = score > self.threshold
            
            if is_fake:
                if current_region is None:
                    # Start new fake region
                    current_region = {
                        'start': segment['start'],
                        'end': segment['end'],
                        'scores': [score]
                    }
                else:
                    # Extend current region
                    current_region['end'] = segment['end']
                    current_region['scores'].append(score)
            else:
                if current_region is not None:
                    # End current region
                    current_region['confidence'] = round(float(np.mean(current_region['scores'])), 4)
                    del current_region['scores']
                    fake_regions.append(current_region)
                    current_region = None
        
        # Add final region if exists
        if current_region is not None:
            current_region['confidence'] = round(float(np.mean(current_region['scores'])), 4)
            del current_region['scores']
            fake_regions.append(current_region)
        
        return fake_regions
    
    def calculate_metrics(self, predicted_regions: List[Dict], ground_truth: Dict) -> Dict:
        """
        Calculate temporal detection metrics against ground truth.
        
        Args:
            predicted_regions: List of predicted fake regions
            ground_truth: Ground truth annotation (from JSON)
            
        Returns:
            Dictionary of metrics:
                - iou: Intersection over Union
                - boundary_precision: Boundary detection accuracy (±2 seconds)
                - segment_accuracy: Per-segment classification accuracy
        """
        # Extract ground truth fake segments
        gt_fake_segments = [s for s in ground_truth['segments'] if s['label'] == 'fake']
        
        if len(gt_fake_segments) == 0:
            return {
                'iou': 0.0,
                'boundary_precision': 0.0,
                'segment_accuracy': 0.0,
                'note': 'No fake segments in ground truth',
                'num_gt_segments': 0,
                'num_pred_regions': len(predicted_regions)
            }
        
        # If no predictions but ground truth has fake segments
        if len(predicted_regions) == 0:
            return {
                'iou': 0.0,
                'boundary_precision': 0.0,
                'segment_accuracy': 0.0,
                'note': 'No fake regions predicted',
                'num_gt_segments': len(gt_fake_segments),
                'num_pred_regions': 0
            }
        
        # Calculate IoU for each ground truth segment
        ious = []
        boundary_matches = []
        
        for gt_seg in gt_fake_segments:
            gt_start, gt_end = gt_seg['start'], gt_seg['end']
            
            # Find best matching predicted region
            best_iou = 0.0
            best_boundary_error = float('inf')
            
            if len(predicted_regions) > 0:
                for pred_reg in predicted_regions:
                    pred_start, pred_end = pred_reg['start'], pred_reg['end']
                    
                    # Calculate IoU
                    intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
                    union = max(gt_end, pred_end) - min(gt_start, pred_start)
                    iou = intersection / union if union > 0 else 0.0
                    
                    if iou > best_iou:
                        best_iou = iou
                        
                        # Calculate boundary error
                        start_error = abs(pred_start - gt_start)
                        end_error = abs(pred_end - gt_end)
                        best_boundary_error = max(start_error, end_error)
            
            ious.append(best_iou)
            
            # Boundary is "correct" if within ±2 seconds
            boundary_matches.append(1 if best_boundary_error <= 2.0 else 0)
        
        avg_iou = float(np.mean(ious)) if ious else 0.0
        avg_boundary = float(np.mean(boundary_matches)) if boundary_matches else 0.0
        
        return {
            'iou': round(avg_iou, 4),
            'boundary_precision': round(avg_boundary, 4),
            'num_gt_segments': len(gt_fake_segments),
            'num_pred_regions': len(predicted_regions)
        }


def load_ground_truth(annotation_path: str) -> Dict:
    """Load ground truth annotation from JSON file."""
    with open(annotation_path, 'r') as f:
        return json.load(f)
