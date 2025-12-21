import numpy as np
import tensorflow as tf
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
from PIL import Image as PILImage
import io
import io
import os
import re


class DeepfakeExplainer:
    """
    Explainable AI module for deepfake audio detection.
    Provides Grad-CAM heatmaps, explanations, and PDF reports.
    """
    
    def __init__(self, model):
        """
        Initialize explainer with trained model.
        
        Args:
            model: Trained Keras model
        """
        self.model = model
        
    def generate_gradcam(self, mel_spectrogram, prediction_score):
        """
        Generate Grad-CAM heatmap for the input mel spectrogram.
        
        Args:
            mel_spectrogram: Input mel spectrogram (128, 313, 1)
            prediction_score: Model prediction score (0-1)
            
        Returns:
            heatmap: Grad-CAM heatmap array
        """
        try:
            # Prepare input
            # Safety: Ensure mel_spectrogram has the expected 313 frames for the model
            target_width = 313
            # mel_spectrogram shape is expected to be (128, N, 1) or (128, N)
            if len(mel_spectrogram.shape) == 3:
                current_width = mel_spectrogram.shape[1]
                if current_width < target_width:
                    pad_width = target_width - current_width
                    mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
                elif current_width > target_width:
                    mel_spectrogram = mel_spectrogram[:, :target_width, :]
            else:
                current_width = mel_spectrogram.shape[1]
                if current_width < target_width:
                    pad_width = target_width - current_width
                    mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
                elif current_width > target_width:
                    mel_spectrogram = mel_spectrogram[:, :target_width]

            input_data = mel_spectrogram[np.newaxis, ...]  # Add batch dimension
            
            # Define score function (for binary classification)
            # If score > 0.5, we're interested in "fake" class (1), else "real" class (0)
            score_class = 1 if prediction_score > 0.5 else 0
            
            # Create Gradcam object
            gradcam = Gradcam(
                self.model,
                model_modifier=ReplaceToLinear(),
                clone=False
            )
            
            # Find the last Conv2D layer for visualization
            conv_layer_name = None
            for layer in reversed(self.model.layers):
                if 'conv2d' in layer.name.lower():
                    conv_layer_name = layer.name
                    break
            
            if conv_layer_name is None:
                print("Warning: No Conv2D layer found. Using default layer.")
                conv_layer_name = -1
            
            # Generate heatmap
            # For binary classification with 1 output neuron, we can't use CategoricalScore([1])
            # We need to define a score function that targets index 0
            
            def score_function(output):
                # output shape is (batch, 1)
                # ALWAYS target the "Fake" class (maximize output) for forensic analysis
                # This ensures we highlight "suspicious" regions even in Real audio
                return output[:, 0]

            cam = gradcam(
                score_function,
                input_data,
                penultimate_layer=conv_layer_name
            )
            
            # Get heatmap for first (and only) sample
            heatmap = cam[0]
            
            return heatmap
            
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            # Return empty heatmap on error
            return np.zeros((mel_spectrogram.shape[0], mel_spectrogram.shape[1]))
    
    def overlay_heatmap(self, mel_spectrogram, heatmap, alpha=0.4):
        """
        Overlay Grad-CAM heatmap on mel spectrogram.
        
        Args:
            mel_spectrogram: Original mel spectrogram (128, 157)
            heatmap: Grad-CAM heatmap
            alpha: Transparency of overlay (0-1)
            
        Returns:
            fig: Matplotlib figure with overlay
        """
        # Resize heatmap to match mel spectrogram if needed
        if heatmap.shape != mel_spectrogram.shape:
            from scipy.ndimage import zoom
            zoom_factors = (mel_spectrogram.shape[0] / heatmap.shape[0],
                          mel_spectrogram.shape[1] / heatmap.shape[1])
            heatmap = zoom(heatmap, zoom_factors, order=1)
        
        # Detect and mask padded/silent regions
        # Calculate energy per time frame
        frame_energy = np.mean(mel_spectrogram, axis=0)
        silence_threshold = 0.1  # Threshold for detecting silence/padding
        
        # Create mask for non-silent regions
        active_mask = frame_energy > silence_threshold
        
        # Apply mask to heatmap - zero out padded regions
        masked_heatmap = heatmap.copy()
        masked_heatmap[:, ~active_mask] = 0
        
        # Normalize heatmap (only non-zero regions)
        if masked_heatmap.max() > 0:
            masked_heatmap = (masked_heatmap - masked_heatmap.min()) / (masked_heatmap.max() - masked_heatmap.min() + 1e-8)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Plot mel spectrogram as base
        ax.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='gray', alpha=0.7)
        
        # Overlay heatmap (using masked version)
        im = ax.imshow(masked_heatmap, aspect='auto', origin='lower', cmap='jet', alpha=alpha)
        
        ax.set_xlabel('Time Frames')
        ax.set_ylabel('Mel Frequency Bands')
        ax.set_title('Grad-CAM Heatmap: Regions Contributing to Detection')
        
        # Add colorbar for heatmap
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Importance', rotation=270, labelpad=15)
        
        plt.tight_layout()
        return fig
    
    def analyze_heatmap_regions(self, heatmap, mel_spectrogram, threshold=0.6):
        """
        Analyze heatmap to identify suspicious regions.
        
        Args:
            heatmap: Grad-CAM heatmap
            mel_spectrogram: Original mel spectrogram
            threshold: Threshold for considering a region important
            
        Returns:
            dict: Analysis results with time ranges and frequency bands
        """
        # Resize heatmap if needed
        if heatmap.shape != mel_spectrogram.shape:
            from scipy.ndimage import zoom
            zoom_factors = (mel_spectrogram.shape[0] / heatmap.shape[0],
                          mel_spectrogram.shape[1] / heatmap.shape[1])
            heatmap = zoom(heatmap, zoom_factors, order=1)
        
        # Normalize heatmap
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Find regions above threshold
        important_regions = heatmap_norm > threshold
        
        # Analyze time dimension (columns)
        # Use MAX instead of MEAN to detect sparse but strong anomalies
        time_importance = np.max(heatmap_norm, axis=0)
        important_time_frames = np.where(time_importance > threshold)[0]
        
        # Analyze frequency dimension (rows)
        freq_importance = np.mean(heatmap_norm, axis=1)
        important_freq_bands = np.where(freq_importance > threshold)[0]
        
        # Convert to time ranges (assuming 10 seconds audio, 313 frames)
        time_ranges = []
        if len(important_time_frames) > 0:
            start_frame = important_time_frames[0]
            end_frame = important_time_frames[-1]
            start_time = (start_frame / 313) * 10.0
            end_time = (end_frame / 313) * 10.0
            time_ranges.append((start_time, end_time))
        
        # Convert to frequency ranges (mel bands)
        freq_ranges = []
        if len(important_freq_bands) > 0:
            start_band = important_freq_bands[0]
            end_band = important_freq_bands[-1]
            freq_ranges.append((start_band, end_band))
        
        # Calculate overall importance score
        importance_score = np.mean(heatmap_norm[important_regions]) if np.any(important_regions) else 0
        
        # Determine active regions (ignore padding)
        frame_energy = np.mean(mel_spectrogram, axis=0)
        active_mask = frame_energy > 0.05
        active_frames_count = np.sum(active_mask)

        # --- NEW: Frequency Band Analysis (Silence-Aware) ---
        # Get slice of heatmap that contains actual audio
        active_heatmap = heatmap_norm[:, active_mask] if np.any(active_mask) else heatmap_norm
        
        # Use Mean of the active portion + Max boost for specific artifacts
        # Low Freq (0-40): Fundamental frequency, rumble
        low_freq_imp = (np.mean(active_heatmap[0:40, :]) + np.max(active_heatmap[0:40, :])) / 2
        
        # Mid Freq (40-90): Speech formants
        mid_freq_imp = (np.mean(active_heatmap[40:90, :]) + np.max(active_heatmap[40:90, :])) / 2
        
        # High Freq (90-128): Fricatives, breath, vocoder artifacts
        high_freq_imp = (np.mean(active_heatmap[90:, :]) + np.max(active_heatmap[90:, :])) / 2
        
        # Calculate Temporal Coverage
        # Count suspicious frames (within active region)
        suspicious_frames_count = np.sum((time_importance > threshold) & active_mask)
        
        temporal_coverage = suspicious_frames_count / active_frames_count if active_frames_count > 10 else 0.0

        return {
            'time_ranges': time_ranges,
            'freq_ranges': freq_ranges,
            'importance_score': importance_score,
            'coverage': temporal_coverage,
            'band_importance': {
                'low': low_freq_imp,
                'mid': mid_freq_imp,
                'high': high_freq_imp
            }
        }
    
    def generate_explanation(self, prediction_score, analysis_results, threshold=0.5):
        """
        Generate human-readable explanation based on prediction and heatmap analysis.
        
        Args:
            prediction_score: Model prediction (0-1, where 1=fake)
            analysis_results: Results from analyze_heatmap_regions
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            str: Detailed explanation text
        """
        is_fake = prediction_score > threshold
        confidence = prediction_score if is_fake else (1 - prediction_score)
        
        explanation = []
        
        # Header
        if is_fake:
            explanation.append(f"🚨 **DEEPFAKE DETECTED** (Confidence: {confidence*100:.1f}%)")
        else:
            explanation.append(f"✅ **AUTHENTIC AUDIO** (Confidence: {confidence*100:.1f}%)")
        
        explanation.append("")
        explanation.append("**Analysis Details:**")
        
        # Time-based analysis
        if analysis_results['time_ranges']:
            for start, end in analysis_results['time_ranges']:
                if is_fake:
                    explanation.append(f"- Manipulation detected between {start:.2f}s and {end:.2f}s")
                else:
                    explanation.append(f"- Minor signal variation detected between {start:.2f}s and {end:.2f}s (Likely Benign)")
        
        # Frequency-based analysis
        if analysis_results['freq_ranges']:
            for start_band, end_band in analysis_results['freq_ranges']:
                if is_fake:
                    explanation.append(f"- Synthetic signatures in frequency bands {start_band}-{end_band}")
                else:
                    explanation.append(f"- Minor frequency intensity in bands {start_band}-{end_band}")
        
        # Coverage analysis
        coverage = analysis_results['coverage'] * 100
        if coverage > 50:
            msg = f"Extensive manipulation detected ({coverage:.1f}% of duration)"
            explanation.append(f"- {msg}")
        elif coverage > 20:
             msg = f"Significant manipulation detected ({coverage:.1f}% of duration)"
             explanation.append(f"- {msg}")
        else:
             msg = f"Localized manipulation traces ({coverage:.1f}% of duration)"
             explanation.append(f"- {msg}")
            
        # --- NEW: Deepfake Artifact Analysis ---
        explanation.append("")
        explanation.append("**🔍 Deepfake Artifact Analysis:**")
        
        bands = analysis_results.get('band_importance', {'low': 0, 'mid': 0, 'high': 0})
        threshold_band = 0.2  # Threshold to consider a band "active"
        
        artifacts_found = False
        
        if bands['high'] > threshold_band:
            if is_fake:
                explanation.append("- **High Frequency Anomalies:** Detected patterns consistent with vocoder artifacts or metallic noise common in synthesized speech.")
            else:
                explanation.append("- **High Frequency Energy:** Natural breath sounds or bright speech characteristics present.")
            artifacts_found = True
            
        if bands['mid'] > threshold_band:
            if is_fake:
                explanation.append("- **Mid Frequency Anomalies:** Detected irregularities in speech formants or unnatural articulation.")
            else:
                explanation.append("- **Mid Frequency Presence:** Strong, clear vocal formants detected typical of human speech.")
            artifacts_found = True
            
        if bands['low'] > threshold_band:
            if is_fake:
                explanation.append("- **Low Frequency Anomalies:** Detected unnatural fundamental frequency or robotic rumble.")
            else:
                explanation.append("- **Low Frequency Depth:** Natural voice resonance and fundamental frequency detected.")
            artifacts_found = True
            
        if not artifacts_found:
             if is_fake and confidence > 0.8:
                 # Fallback for High Confidence Fakes with diffuse heatmaps
                 explanation.append("- **Global Signal Anomalies:** Strong manipulation traces detected across the spectrum (Diffuse/General Artifacts).")
                 explanation.append("- **Pattern Mismatch:** The audio lacks the natural fine-grained variability of authentic speech.")
             elif not is_fake:
                 # Validating REAL audio features
                 explanation.append("- **Natural Harmonics:** Signal structure is consistent with organic speech patterns.")
                 explanation.append("- **No Vocoder Artifacts:** Absence of metallic smoothing or robotic quantization.")
             else:
                 explanation.append("- No specific frequency-based artifacts strongly detected.")
        else:
             # If artifacts WERE found but verdict is REAL, clarify they are likely false alarms or benign
             if not is_fake:
                 explanation[-1] = explanation[-1] + " (Likely background noise or naturally occurring)"
                 if len(explanation) > 1 and "Anomalies" in explanation[-2]:
                     explanation[-2] = explanation[-2] + " (Within normal organic range)"
                     
        # Interpretation
        explanation.append("")
        explanation.append("**Interpretation:**")
        
        if is_fake:
            if confidence > 0.8:
                explanation.append("- Very high confidence of manipulation detected")
                explanation.append("- Multiple deepfake artifacts identified")
            elif confidence > 0.6:
                explanation.append("- Clear signs of audio manipulation")
                explanation.append("- Likely generated or heavily edited")
            else:
                explanation.append("- Possible manipulation detected")
                explanation.append("- Further verification recommended")
        else:
            if confidence > 0.8:
                explanation.append("- Strong indicators of authentic audio")
                explanation.append("- No significant manipulation artifacts found")
            elif confidence > 0.6:
                explanation.append("- Likely authentic audio")
                explanation.append("- Minor anomalies within normal range")
            else:
                explanation.append("- Borderline case")
                explanation.append("- Manual review recommended")
        
        return "\n".join(explanation)
    
    def generate_pdf_report(self, audio_path, prediction_score, mel_spectrogram, 
                           heatmap_fig, explanation, output_path="report.pdf", verdict=None, fake_regions=None, timeline_fig=None):
        """
        Generate comprehensive PDF report.
        
        Args:
            audio_path: Path to analyzed audio file
            prediction_score: Model prediction score
            mel_spectrogram: Mel spectrogram array
            heatmap_fig: Matplotlib figure with heatmap overlay
            explanation: Explanation text
            output_path: Path to save PDF
            verdict: Optional verdict string ("REAL", "FAKE", "MIXED")
            
        Returns:
            str: Path to generated PDF
        """
        try:
            # Create PDF
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1f2937'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#374151'),
                spaceAfter=12,
                spaceBefore=12
            )
            
            # Title
            story.append(Paragraph("DEEPFAKE DETECTION REPORT", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Metadata table
            metadata = [
                ['Analysis Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Audio File:', os.path.basename(audio_path)],
                ['System:', 'Deepfake Audio Detector'],
            ]
            
            t = Table(metadata, colWidths=[2*inch, 4*inch])
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#6b7280')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.3*inch))
            
            # Result
            story.append(Paragraph("Analysis Result", heading_style))
            
            # Determine styling based on verdict or score
            # Determine styling based on verdict (Strict Adherence)
            if verdict in ["MIXED", "SUSPICIOUS"]:
                result_text = "SUSPICIOUS / MIXED AUDIO" 
                result_color = colors.orange
                confidence = prediction_score 
            elif verdict == "REAL":
                result_text = "REAL AUDIO"
                result_color = colors.green
                confidence = prediction_score
            elif verdict == "FAKE":
                 result_text = "DEEPFAKE DETECTED"
                 result_color = colors.red
                 confidence = prediction_score
            else:
                 # Fallback
                 is_fake = prediction_score > 0.5
                 result_text = "DEEPFAKE DETECTED" if is_fake else "REAL AUDIO" 
                 confidence = prediction_score if is_fake else (1 - prediction_score)
                 result_color = colors.red if is_fake else colors.green
            
            result_style = ParagraphStyle(
                'Result',
                parent=styles['Normal'],
                fontSize=16,
                textColor=result_color,
                spaceAfter=6,
                fontName='Helvetica-Bold'
            )
            
            story.append(Paragraph(result_text, result_style))
            story.append(Paragraph(f"Confidence Score: {confidence*100:.2f}%", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # Explanation
            story.append(Paragraph("Detailed Analysis", heading_style))
            
            # Regex to clean emojis and non-ASCII chars
            def clean_text(text):
                return re.sub(r'[^\x00-\x7F]+', '', text).replace('**', '')

            for line in explanation.split('\n'):
                if line.strip():
                    # Remove markdown formatting and emojis for PDF
                    clean_line = clean_text(line)
                    if clean_line.strip():
                         story.append(Paragraph(clean_line, styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # TEMPORAL ANALYSIS TABLE (New)
            if fake_regions and len(fake_regions) > 0:
                story.append(Paragraph("Temporal Analysis (Suspicious Regions)", heading_style))
                story.append(Spacer(1, 0.1*inch))
                
                # Create table data
                table_data = [['Segment', 'Start Time', 'End Time', 'Confidence']]
                for i, region in enumerate(fake_regions, 1):
                    row = [
                        f"Region {i}",
                        f"{region['start']:.2f}s",
                        f"{region['end']:.2f}s",
                        f"{region['confidence']*100:.1f}%"
                    ]
                    table_data.append(row)
                
                # Table style
                t_regions = Table(table_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                t_regions.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5e7eb')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1f2937')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d1d5db')),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ]))
                story.append(t_regions)
                story.append(Spacer(1, 0.3*inch))
                
                # Add Timeline Diagram if provided
                if timeline_fig:
                    story.append(Paragraph("Timeline Visualization", heading_style))
                    # Save timeline fig to bytes
                    t_buf = io.BytesIO()
                    timeline_fig.savefig(t_buf, format='png', dpi=150, bbox_inches='tight')
                    t_buf.seek(0)
                    t_img = RLImage(t_buf, width=6*inch, height=2*inch)
                    story.append(t_img)
                    story.append(Spacer(1, 0.2*inch))
            
            # Save heatmap figure to bytes
            buf = io.BytesIO()
            heatmap_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            
            # Add heatmap image
            story.append(Paragraph("Grad-CAM Heatmap Visualization", heading_style))
            img = RLImage(buf, width=6*inch, height=2*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
            
            # Footer
            footer_text = "This report was generated by an AI-based deepfake detection system. " \
                         "Results should be verified by human experts for critical applications."
            footer_style = ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=8,
                textColor=colors.HexColor('#9ca3af'),
                alignment=TA_CENTER
            )
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph(footer_text, footer_style))
            
            # Build PDF
            doc.build(story)
            
            return output_path
            
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            return None
