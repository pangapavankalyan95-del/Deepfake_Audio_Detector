"""
Temporal Validation Script
Tests temporal analysis on validation dataset and calculates metrics.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.temporal_analyzer import TemporalAnalyzer, load_ground_truth
from src.temporal_visualizer import TemporalVisualizer
from tensorflow.keras.models import load_model

def find_latest_model():
    """Find the latest trained model."""
    model_dir = Path("models")
    
    # Look for timestamped model directories
    model_subdirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('model_')]
    if model_subdirs:
        latest_dir = max(model_subdirs, key=lambda x: x.stat().st_mtime)
        model_path = latest_dir / "model.keras"
        if model_path.exists():
            return str(model_path)
    
    # Fallback to direct .keras files
    model_files = list(model_dir.glob("*.keras"))
    if model_files:
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        return str(latest_model)
    
    return None

def validate_temporal_detection(num_samples=10, visualize=True):
    """
    Validate temporal detection on dataset.
    
    Args:
        num_samples: Number of samples to validate
        visualize: Whether to create visualizations
    """
    print("="*70)
    print("TEMPORAL DEEPFAKE DETECTION VALIDATION")
    print("="*70)
    
    # Load model
    print("\n[1/5] Loading model...")
    model_path = find_latest_model()
    if not model_path:
        print("ERROR: No model found!")
        return
    
    print(f"Loading: {model_path}")
    model = load_model(model_path)
    print("✓ Model loaded")
    
    # Initialize analyzer and visualizer
    print("\n[2/5] Initializing temporal analyzer...")
    analyzer = TemporalAnalyzer(model, window_size=5.0, overlap=0.5, threshold=0.3)
    visualizer = TemporalVisualizer(dark_mode=True)
    print("✓ Analyzer initialized")
    
    # Load validation dataset
    print("\n[3/5] Loading validation dataset...")
    mixed_audio_dir = Path("data/temporal_validation/mixed_audio")
    annotations_dir = Path("data/temporal_validation/annotations")
    
    audio_files = list(mixed_audio_dir.glob("*.wav"))
    if len(audio_files) == 0:
        print("ERROR: No validation samples found!")
        return
    
    print(f"Found {len(audio_files)} validation samples")
    
    # Select samples to validate
    import random
    random.seed(42)
    samples_to_validate = random.sample(audio_files, min(num_samples, len(audio_files)))
    print(f"Validating {len(samples_to_validate)} samples...")
    
    # Validate each sample
    print("\n[4/5] Running temporal analysis...")
    results = []
    all_metrics = []
    
    for i, audio_file in enumerate(samples_to_validate, 1):
        print(f"\n  Sample {i}/{len(samples_to_validate)}: {audio_file.name}")
        
        # Find corresponding annotation
        annotation_file = annotations_dir / audio_file.name.replace('.wav', '.json')
        if not annotation_file.exists():
            print(f"    Warning: No annotation found, skipping...")
            continue
        
        # Load ground truth
        ground_truth = load_ground_truth(str(annotation_file))
        
        # Run temporal analysis
        try:
            analysis_result = analyzer.analyze_temporal(str(audio_file), apply_noise_reduction=True)
            
            # Calculate metrics
            metrics = analyzer.calculate_metrics(
                analysis_result.get('fake_regions', []),
                ground_truth
            )
            
            all_metrics.append(metrics)
            
            # Print results
            print(f"    Duration: {analysis_result['duration']:.1f}s")
            print(f"    Segments: {analysis_result['num_segments']}")
            print(f"    Fake regions detected: {len(analysis_result.get('fake_regions', []))}")
            print(f"    IoU: {metrics.get('iou', 0)*100:.1f}%")
            print(f"    Boundary Precision: {metrics.get('boundary_precision', 0)*100:.1f}%")
            
            # Store result
            results.append({
                'filename': audio_file.name,
                'analysis': analysis_result,
                'ground_truth': ground_truth,
                'metrics': metrics
            })
            
            # Create visualization if requested
            if visualize and i <= 3:  # Only first 3 samples
                print(f"    Creating visualization...")
                output_dir = Path("data/temporal_validation/visualizations")
                output_dir.mkdir(exist_ok=True)
                
                # Timeline plot
                fig = visualizer.plot_timeline(analysis_result, ground_truth)
                fig.savefig(output_dir / f"{audio_file.stem}_timeline.png", 
                           dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
                plt.close(fig)
                
                # Waveform with timeline
                fig = visualizer.plot_waveform_with_timeline(str(audio_file), analysis_result)
                fig.savefig(output_dir / f"{audio_file.stem}_waveform.png",
                           dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
                plt.close(fig)
                
                print(f"    ✓ Visualizations saved")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
    
    # Calculate overall metrics
    print("\n[5/5] Calculating overall metrics...")
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    if all_metrics:
        avg_iou = np.mean([m.get('iou', 0) for m in all_metrics])
        avg_boundary_precision = np.mean([m.get('boundary_precision', 0) for m in all_metrics])
        
        print(f"\nSamples validated: {len(all_metrics)}")
        print(f"Average IoU: {avg_iou*100:.1f}%")
        print(f"Average Boundary Precision (±2s): {avg_boundary_precision*100:.1f}%")
        
        # Save results
        results_file = Path("data/temporal_validation/validation_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'num_samples': len(all_metrics),
                'avg_iou': float(avg_iou),
                'avg_boundary_precision': float(avg_boundary_precision),
                'individual_results': results
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_file}")
        
        if visualize:
            print(f"✓ Visualizations saved to: data/temporal_validation/visualizations/")
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    # Run validation on ALL samples
    validate_temporal_detection(num_samples=50, visualize=True)
