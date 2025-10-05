# plot.py

import os
import torch
from torchvision import utils as vutils
import matplotlib.pyplot as plt

def plot_history(train_loss, val_loss, train_mae, val_mae, save_path):
    """Plots and saves the training and validation history for loss and MAE."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    fig.suptitle('Training Performance Metrics', fontsize=16)

    # Plotting Loss Mean SSquare Error (MSE)
    ax1.plot(train_loss, label='Training Loss')
    ax1.plot(val_loss, label='Validation Loss')
    ax1.set_title('Mean Squared Error (MSE) Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plotting Mean Absolute Error (MAE)
    ax2.plot(train_mae, label='Training MAE')
    ax2.plot(val_mae, label='Validation MAE')
    ax2.set_title('Mean Absolute Error (MAE)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close() # Close the figure to free up memory

def save_validation_comparison(model, dataloader, device, save_path):
    """Generates and saves a visual comparison for a validation batch."""
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(dataloader))
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        
        last_sat_input = inputs[:, -1, :, :, :]
        ground_truth = targets[:, 0, :, :, :]
        prediction = outputs[:, 0, :, :, :]
        
        last_sat_input = torch.clamp(last_sat_input, 0, 1)
        ground_truth = torch.clamp(ground_truth, 0, 1)
        prediction = torch.clamp(prediction, 0, 1)
        
        if ground_truth.shape[1] == 1:
            ground_truth = ground_truth.repeat(1, 3, 1, 1)
        if prediction.shape[1] == 1:
            prediction = prediction.repeat(1, 3, 1, 1)
            
        comparison_tensor = torch.stack([last_sat_input, prediction, ground_truth], dim=1).flatten(0, 1)
        
        vutils.save_image(comparison_tensor, save_path, nrow=3, normalize=False)
        print(f"Validation comparison image saved to {save_path}")