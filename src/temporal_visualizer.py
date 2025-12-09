"""
Temporal Visualization Module
Creates timeline visualizations for temporal deepfake detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Optional
import librosa
import librosa.display

class TemporalVisualizer:
    """
    Creates visualizations for temporal analysis results.
    """
    
    def __init__(self, dark_mode=True):
        """
        Initialize visualizer.
        
        Args:
            dark_mode: Use dark theme for plots
        """
        self.dark_mode = dark_mode
        if dark_mode:
            plt.style.use('dark_background')
    
    def plot_timeline(self, analysis_result: Dict, ground_truth: Optional[Dict] = None, 
                     figsize=(12, 4)) -> plt.Figure:
        """
        Plot timeline showing real/fake segments.
        
        Args:
            analysis_result: Result from TemporalAnalyzer.analyze_temporal()
            ground_truth: Optional ground truth annotation
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        duration = analysis_result['duration']
        segments = analysis_result['segments']
        
        # Plot segments as colored bars
        if not segments:
             # If no segments (e.g. short audio or error), plot a single 'Unknown' or 'Real' bar
            rect = patches.Rectangle((0, 0), duration, 1, linewidth=0, facecolor='green', alpha=0.3)
            ax.add_patch(rect)
            ax.text(duration/2, 0.5, "NO ANOMALIES DETECTED", ha='center', va='center', color='white', fontsize=12)
        else:
            for segment in segments:
                start = segment['start']
                end = segment['end']
                score = segment['score']
                is_fake = segment['label'] == 'fake'
                
                # Color based on prediction
                if is_fake:
                    color = plt.cm.Reds(score)  # Red gradient for fake
                    alpha = 0.8
                else:
                    color = plt.cm.Greens(1 - score)  # Green gradient for real
                    alpha = 0.6
                
                # Draw rectangle
                rect = patches.Rectangle(
                    (start, 0), end - start, 1,
                    linewidth=0, facecolor=color, alpha=alpha
                )
                ax.add_patch(rect)
        
        # Plot ground truth if available
        if ground_truth:
            for gt_seg in ground_truth['segments']:
                start = gt_seg['start']
                end = gt_seg['end']
                label = gt_seg['label']
                
                # Draw ground truth as outline
                color = 'red' if label == 'fake' else 'green'
                rect = patches.Rectangle(
                    (start, 1.1), end - start, 0.2,
                    linewidth=2, edgecolor=color, facecolor='none', linestyle='--'
                )
                ax.add_patch(rect)
        
        # Formatting
        ax.set_xlim(0, duration)
        ax.set_ylim(0, 1.5 if ground_truth else 1)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Prediction', fontsize=12)
        ax.set_yticks([0.5] + ([1.2] if ground_truth else []))
        ax.set_yticklabels(['Detected'] + (['Ground Truth'] if ground_truth else []))
        ax.set_title('Temporal Deepfake Detection Timeline', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.8, label='Real'),
            Patch(facecolor='red', alpha=0.8, label='Fake')
        ]
        if ground_truth:
            legend_elements.append(Patch(edgecolor='white', facecolor='none', 
                                        linestyle='--', label='Ground Truth'))
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def plot_confidence_heatmap(self, analysis_result: Dict, figsize=(12, 3)) -> plt.Figure:
        """
        Plot confidence heatmap over time.
        
        Args:
            analysis_result: Result from TemporalAnalyzer.analyze_temporal()
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        segments = analysis_result['segments']
        
        # Extract data
        times = [(s['start'] + s['end']) / 2 for s in segments]
        scores = [s['score'] for s in segments]
        
        # Create heatmap
        scatter = ax.scatter(times, [0.5] * len(times), c=scores, 
                           cmap='RdYlGn_r', s=200, marker='s', 
                           vmin=0, vmax=1, alpha=0.8)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.2)
        cbar.set_label('Fake Probability', fontsize=11)
        
        # Formatting
        ax.set_xlim(0, analysis_result['duration'])
        ax.set_ylim(0, 1)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_yticks([])
        ax.set_title('Confidence Heatmap', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def plot_waveform_with_timeline(self, audio_path: str, analysis_result: Dict,
                                    figsize=(14, 6)) -> plt.Figure:
        """
        Plot waveform with fake regions highlighted.
        
        Args:
            audio_path: Path to audio file
            analysis_result: Result from TemporalAnalyzer.analyze_temporal()
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot waveform
        times = np.arange(len(y)) / sr
        ax1.plot(times, y, color='cyan', alpha=0.7, linewidth=0.5)
        ax1.set_ylabel('Amplitude', fontsize=11)
        ax1.set_title('Audio Waveform with Detected Fake Regions', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Highlight fake regions on waveform
        for region in analysis_result.get('fake_regions', []):
            ax1.axvspan(region['start'], region['end'], 
                       color='red', alpha=0.3, label='Fake Region')
        
        # Remove duplicate labels
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Plot timeline
        duration = analysis_result['duration']
        segments = analysis_result['segments']
        
        for segment in segments:
            start = segment['start']
            end = segment['end']
            is_fake = segment['label'] == 'fake'
            color = 'red' if is_fake else 'green'
            
            rect = patches.Rectangle(
                (start, 0), end - start, 1,
                linewidth=0, facecolor=color, alpha=0.7
            )
            ax2.add_patch(rect)
        
        ax2.set_xlim(0, duration)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Detection', fontsize=11)
        ax2.set_yticks([0.5])
        ax2.set_yticklabels(['Timeline'])
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def plot_segment_scores(self, analysis_result: Dict, figsize=(12, 4)) -> plt.Figure:
        """
        Plot segment-by-segment scores as bar chart.
        
        Args:
            analysis_result: Result from TemporalAnalyzer.analyze_temporal()
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        segments = analysis_result['segments']
        
        # Extract data
        segment_labels = [f"{s['start']:.1f}-{s['end']:.1f}s" for s in segments]
        scores = [s['score'] for s in segments]
        colors = ['red' if s['label'] == 'fake' else 'green' for s in segments]
        
        # Create bar chart
        bars = ax.bar(range(len(scores)), scores, color=colors, alpha=0.7, edgecolor='white')
        
        # Add threshold line
        ax.axhline(y=0.5, color='yellow', linestyle='--', linewidth=2, label='Threshold')
        
        # Formatting
        ax.set_xlabel('Segment', fontsize=12)
        ax.set_ylabel('Fake Probability', fontsize=12)
        ax.set_title('Segment-by-Segment Analysis', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels(segment_labels, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_summary_report(self, analysis_result: Dict, ground_truth: Optional[Dict] = None,
                             metrics: Optional[Dict] = None) -> str:
        """
        Create text summary of analysis results.
        
        Args:
            analysis_result: Result from TemporalAnalyzer.analyze_temporal()
            ground_truth: Optional ground truth annotation
            metrics: Optional metrics dictionary
            
        Returns:
            Formatted summary string
        """
        summary = []
        summary.append("=" * 60)
        summary.append("TEMPORAL DEEPFAKE DETECTION SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Duration: {analysis_result['duration']:.2f} seconds")
        summary.append(f"Number of segments analyzed: {analysis_result['num_segments']}")
        summary.append(f"Window size: {analysis_result['window_size']:.1f}s")
        summary.append(f"Overlap: {analysis_result['overlap']*100:.0f}%")
        summary.append(f"Overall fake probability: {analysis_result['overall_score']*100:.1f}%")
        summary.append("")
        
        # Fake regions
        fake_regions = analysis_result.get('fake_regions', [])
        if fake_regions:
            summary.append(f"DETECTED FAKE REGIONS: {len(fake_regions)}")
            summary.append("-" * 60)
            for i, region in enumerate(fake_regions, 1):
                summary.append(f"  Region {i}: {region['start']:.2f}s - {region['end']:.2f}s "
                             f"(Confidence: {region['confidence']*100:.1f}%)")
        else:
            summary.append("NO FAKE REGIONS DETECTED")
        
        summary.append("")
        
        # Metrics if available
        if metrics:
            summary.append("VALIDATION METRICS")
            summary.append("-" * 60)
            summary.append(f"  IoU (Intersection over Union): {metrics.get('iou', 0)*100:.1f}%")
            summary.append(f"  Boundary Precision (±2s): {metrics.get('boundary_precision', 0)*100:.1f}%")
            summary.append(f"  Ground truth segments: {metrics.get('num_gt_segments', 0)}")
            summary.append(f"  Predicted regions: {metrics.get('num_pred_regions', 0)}")
        
        summary.append("=" * 60)
        
        return "\n".join(summary)
