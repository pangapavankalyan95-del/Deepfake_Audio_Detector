import pandas as pd
import matplotlib.pyplot as plt
import os

# --- PATH TO YOUR LOG FILE ---
log_path = r'models\model_20251216_102910\training_log.csv'
output_path = r'models\model_20251216_102910\training_history_full.png'

if not os.path.exists(log_path):
    print(f"Error: Could not find {log_path}")
else:
    # Load the data
    df = pd.read_csv(log_path)
    
    # Sort by epoch just in case resumption added them out of order
    df = df.sort_values('epoch')
    
    # Create the figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3)
    
    # Metrics to plot
    metrics = {
        'accuracy': 'Accuracy',
        'loss': 'Loss',
        'precision': 'Precision',
        'recall': 'Recall'
    }
    
    # Find the actual precision/recall column names (they vary based on TF version)
    cols = df.columns
    prec_col = [c for c in cols if 'precision' in c.lower()][0]
    rec_col = [c for c in cols if 'recall' in c.lower()][0]
    
    mapping = {
        'accuracy': ('accuracy', 'val_accuracy', axes[0, 0]),
        'loss': ('loss', 'val_loss', axes[0, 1]),
        'precision': (prec_col, f'val_{prec_col}', axes[1, 0]),
        'recall': (rec_col, f'val_{rec_col}', axes[1, 1])
    }
    
    for label, (train_key, val_key, ax) in mapping.items():
        ax.plot(df['epoch'], df[train_key], label='Train', linewidth=2)
        ax.plot(df['epoch'], df[val_key], label='Val', linewidth=2)
        ax.set_title(f'Full {label.capitalize()} History (37 Epochs)', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Success! Full history graph saved to: {output_path}")
    plt.show()
