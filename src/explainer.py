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
import os


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
            mel_spectrogram: Input mel spectrogram (128, 157, 1)
            prediction_score: Model prediction score (0-1)
            
        Returns:
            heatmap: Grad-CAM heatmap array
        """
        try:
            # Prepare input
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
            cam = gradcam(
                CategoricalScore([score_class]),
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
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Display mel spectrogram
        im1 = ax.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
        
        # Overlay heatmap
        im2 = ax.imshow(heatmap, aspect='auto', origin='lower', cmap='jet', alpha=alpha)
        
        ax.set_xlabel('Time Frames')
        ax.set_ylabel('Mel Frequency Bands')
        ax.set_title('Grad-CAM Heatmap: Regions Contributing to Detection')
        
        # Add colorbar for heatmap
        cbar = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
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
        time_importance = np.mean(heatmap_norm, axis=0)
        important_time_frames = np.where(time_importance > threshold)[0]
        
        # Analyze frequency dimension (rows)
        freq_importance = np.mean(heatmap_norm, axis=1)
        important_freq_bands = np.where(freq_importance > threshold)[0]
        
        # Convert to time ranges (assuming 5 seconds audio, 157 frames)
        time_ranges = []
        if len(important_time_frames) > 0:
            start_frame = important_time_frames[0]
            end_frame = important_time_frames[-1]
            start_time = (start_frame / 157) * 5.0
            end_time = (end_frame / 157) * 5.0
            time_ranges.append((start_time, end_time))
        
        # Convert to frequency ranges (mel bands)
        freq_ranges = []
        if len(important_freq_bands) > 0:
            start_band = important_freq_bands[0]
            end_band = important_freq_bands[-1]
            freq_ranges.append((start_band, end_band))
        
        # Calculate overall importance score
        importance_score = np.mean(heatmap_norm[important_regions]) if np.any(important_regions) else 0
        
        return {
            'time_ranges': time_ranges,
            'freq_ranges': freq_ranges,
            'importance_score': importance_score,
            'coverage': np.sum(important_regions) / important_regions.size
        }
    
    def generate_explanation(self, prediction_score, analysis_results):
        """
        Generate human-readable explanation based on prediction and heatmap analysis.
        
        Args:
            prediction_score: Model prediction (0-1, where 1=fake)
            analysis_results: Results from analyze_heatmap_regions
            
        Returns:
            str: Detailed explanation text
        """
        is_fake = prediction_score > 0.5
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
                explanation.append(f"- Suspicious activity detected between {start:.2f}s and {end:.2f}s")
        
        # Frequency-based analysis
        if analysis_results['freq_ranges']:
            for start_band, end_band in analysis_results['freq_ranges']:
                explanation.append(f"- Anomalies found in mel frequency bands {start_band}-{end_band}")
        
        # Coverage analysis
        coverage = analysis_results['coverage'] * 100
        if coverage > 50:
            explanation.append(f"- High coverage of suspicious regions ({coverage:.1f}% of spectrogram)")
        elif coverage > 20:
            explanation.append(f"- Moderate coverage of suspicious regions ({coverage:.1f}% of spectrogram)")
        else:
            explanation.append(f"- Low coverage of suspicious regions ({coverage:.1f}% of spectrogram)")
        
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
                           heatmap_fig, explanation, output_path="report.pdf"):
        """
        Generate comprehensive PDF report.
        
        Args:
            audio_path: Path to analyzed audio file
            prediction_score: Model prediction score
            mel_spectrogram: Mel spectrogram array
            heatmap_fig: Matplotlib figure with heatmap overlay
            explanation: Explanation text
            output_path: Path to save PDF
            
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
            story.append(Paragraph("Deepfake Audio Detection Report", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Metadata table
            metadata = [
                ['Analysis Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Audio File:', os.path.basename(audio_path)],
                ['Model:', 'CNN + BiLSTM Deepfake Detector'],
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
            story.append(Paragraph("Detection Result", heading_style))
            is_fake = prediction_score > 0.5
            result_text = "DEEPFAKE DETECTED" if is_fake else "AUTHENTIC AUDIO"
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
            story.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # Explanation
            story.append(Paragraph("Detailed Analysis", heading_style))
            for line in explanation.split('\n'):
                if line.strip():
                    # Remove markdown formatting for PDF
                    clean_line = line.replace('**', '').replace('🚨', '').replace('✅', '')
                    story.append(Paragraph(clean_line, styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
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
