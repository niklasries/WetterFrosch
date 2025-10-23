# plot.py

import os
import torch # type: ignore
from torchvision import utils as vutils # type: ignore
import matplotlib.pyplot as plt # type: ignore

def plot_history(history_dict, save_path):
    """
    Plots and saves the training history for all provided metrics.
    Expects history_dict in the format: {'metric_name': (train_history, val_history)}
    """
    metrics = list(history_dict.keys())
    num_metrics = len(metrics)
    
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 6 * num_metrics))
    if num_metrics == 1:
        axes = [axes]
        
    fig.suptitle('Training Performance Metrics', fontsize=16)

    for i, metric in enumerate(metrics):
        train_history, val_history = history_dict[metric]
        ax = axes[i]
        
        ax.plot(train_history, label=f'Training {metric.capitalize()}')
        ax.plot(val_history, label=f'Validation {metric.capitalize()}')
        ax.set_title(f'{metric.capitalize()} History')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)
        
        # Set y-axis limits for metrics that are bounded between 0 and 1
        if metric in ['f1', 'precision', 'recall']:
            ax.set_ylim([0, 1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()


def save_validation_comparison(model, dataloader, device, save_path):
    """
    Generates and saves a visual comparison for a validation batch, showing the 
    full predicted sequence against the ground truth for each sample.
    """
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(dataloader))
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        
        # (B, C, H, W) - The last satellite image used as input
        last_sat_input = inputs[:, -1, :, :, :]
        
        # (B, T, C, H, W) - The full sequence for ground truth and prediction
        ground_truth_seq = targets
        prediction_seq = outputs
        
        if last_sat_input.shape[1] == 1:
            last_sat_input = last_sat_input.repeat(1, 3, 1, 1)
        if ground_truth_seq.shape[2] == 1:
            ground_truth_seq = ground_truth_seq.repeat(1, 1, 3, 1, 1)
        if prediction_seq.shape[2] == 1:
            prediction_seq = prediction_seq.repeat(1, 1, 3, 1, 1)
        
        num_frames = prediction_seq.shape[1]
        
        interleaved_seq = torch.stack([ground_truth_seq, prediction_seq], dim=2).flatten(1, 2)
        full_comparison = torch.cat([last_sat_input.unsqueeze(1), interleaved_seq], dim=1)
        comparison_tensor = full_comparison.flatten(0, 1)
        
        vutils.save_image(comparison_tensor, save_path, nrow=1 + 2 * num_frames, normalize=False)
        print(f"Validation comparison image saved to {save_path}")