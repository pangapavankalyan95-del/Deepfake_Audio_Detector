"""
Forensic Report Generator Module
Generates professional PDF reports for deepfake audio analysis.
"""

import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io

class ForensicReportGenerator:
    """Generates PDF reports for forensic analysis."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        
    def _create_custom_styles(self):
        """Create custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='Header',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#00F0FF'),
            spaceAfter=30,
            alignment=1  # Center
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#0078FF'),
            spaceBefore=20,
            spaceAfter=10
        ))
        
        self.styles.add(ParagraphStyle(
            name='VerdictReal',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=colors.green,
            alignment=1,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='VerdictFake',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=colors.red,
            alignment=1,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='VerdictMixed',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=colors.orange,
            alignment=1,
            fontName='Helvetica-Bold'
        ))

    def generate_report(self, 
                       audio_filename: str,
                       prediction_score: float,
                       threshold: float,
                       speaker_id: str,
                       speaker_conf: float,
                       explanation: str = None,
                       plot_images: dict = None,
                       temporal_result: dict = None,
                       verdict: str = None) -> bytes:
        """
        Generate PDF report.
        
        Args:
            audio_filename: Name of the analyzed file
            prediction_score: Fake probability (0-1)
            threshold: Detection threshold
            speaker_id: Identified speaker name
            speaker_conf: Speaker identification confidence
            plot_images: Dictionary of plot images (bytes or paths)
            temporal_result: Optional temporal analysis results
            verdict: Optional verdict string (REAL, FAKE, MIXED)
            
        Returns:
            PDF bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        
        # 1. Header
        story.append(Paragraph("DEEPFAKE AUDIO FORENSIC REPORT", self.styles['Header']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # 2. Verdict Section
        if verdict == "MIXED":
             verdict_text = "VERDICT: SUSPICIOUS / MIXED AUDIO"
             verdict_style = self.styles['VerdictMixed']
        else:
             is_fake = prediction_score > threshold
             verdict_text = "VERDICT: FAKE AUDIO DETECTED" if is_fake else "VERDICT: AUTHENTIC AUDIO CONFIRMED"
             verdict_style = self.styles['VerdictFake'] if is_fake else self.styles['VerdictReal']
        
        story.append(Paragraph(verdict_text, verdict_style))
        story.append(Spacer(1, 20))
        
        # 3. Analysis Details Table
        # Determine speaker match display
        if speaker_id and speaker_id not in ["Unknown", "UNKNOWN", "NO MATCH", ""]:
            speaker_match_value = f"{speaker_conf*100:.2f}%"
        else:
            speaker_match_value = "No Match"
            speaker_id = "Unknown"
        
        data = [
            ['Parameter', 'Value'],
            ['File Name', audio_filename],
            ['Fake Probability', f"{prediction_score*100:.2f}%"],
            ['Confidence', f"{max(prediction_score, 1-prediction_score)*100:.2f}%"],
            ['Speaker Identity', speaker_id],
            ['Speaker Match', speaker_match_value]
        ]
        
        t = Table(data, colWidths=[2.5*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 20))

        # 3.5. XAI Explanation (NEW)
        if explanation:
            story.append(Paragraph("Detailed XAI Analysis", self.styles['SubHeader']))
            # Process explanation text (remove markdown mostly)
            for line in explanation.split('\n'):
                if line.strip():
                    clean_line = line.replace('**', '').replace('###', '').replace('🚨', '').replace('✅', '').replace('⚠️', '')
                    story.append(Paragraph(clean_line, self.styles['Normal']))
            story.append(Spacer(1, 20))
        
        # 4. Visualizations
        if plot_images:
            story.append(Paragraph("Visual Analysis", self.styles['SubHeader']))
            
            # Mel Spectrogram
            if 'mel_spec' in plot_images:
                story.append(Paragraph("Mel Spectrogram Analysis", self.styles['Heading3']))
                img = Image(plot_images['mel_spec'], width=6*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 10))
                
            # Radar Chart
            if 'radar' in plot_images:
                story.append(Paragraph("Frequency Signature", self.styles['Heading3']))
                img = Image(plot_images['radar'], width=4*inch, height=4*inch)
                story.append(img)
            
            # Temporal Timeline
            if 'timeline' in plot_images:
                story.append(Spacer(1, 10))
                story.append(Paragraph("Temporal Analysis Timeline", self.styles['Heading3']))
                img = Image(plot_images['timeline'], width=6*inch, height=2.5*inch)
                story.append(img)
        
        # 4.5 Temporal Analysis Results
        if temporal_result:
            story.append(Spacer(1, 20))
            story.append(Paragraph("Temporal Analysis", self.styles['SubHeader']))
            
            fake_regions = temporal_result.get('fake_regions', [])
            
            if fake_regions:
                story.append(Paragraph(f"Detected {len(fake_regions)} fake segment(s):", self.styles['Normal']))
                story.append(Spacer(1, 10))
                
                # Create table of fake segments
                temporal_data = [['Region', 'Start Time', 'End Time', 'Confidence']]
                for i, region in enumerate(fake_regions, 1):
                    temporal_data.append([
                        f"Region {i}",
                        f"{region['start']:.2f}s",
                        f"{region['end']:.2f}s",
                        f"{region['confidence']*100:.1f}%"
                    ])
                
                t_temporal = Table(temporal_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                t_temporal.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (3, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (3, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(t_temporal)
            else:
                story.append(Paragraph("No fake segments detected - Audio appears authentic throughout.", 
                                     self.styles['Normal']))
        
        # 5. Disclaimer
        story.append(Spacer(1, 30))
        disclaimer = ("DISCLAIMER: This report is generated by an AI-based deepfake detection system. "
                     "Results should be verified by human experts for legal or critical applications.")
        story.append(Paragraph(disclaimer, self.styles['Italic']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
