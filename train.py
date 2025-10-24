# train.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The video decoding*")

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader, random_split # type: ignore
from torchvision import transforms # type: ignore

import os
from tqdm import tqdm # type: ignore
import shutil

from plot import plot_history, save_validation_comparison

from utils import WeatherDataset, WeightedBCELoss
from utils import calculate_binary_metrics
from model import SpatioTemporalTransformer

SEED = 42
import random
import numpy as np
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

torch.set_float32_matmul_precision('high')

DATA_DIR  = "/wetter/input/WetterDaten/"
#pre generated video cache under 
#CACHE_DIR = "/wetter/input/WetterDatenCacheSmol/"
CACHE_DIR = "/wetter/input/WetterDatenCache/"


LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 16
EPOCHS = 120*2

VALIDATION_SPLIT = 0.2 

WINDOW_SIZE = 8       # 12 steps = 3 hours
PREDICTION_STEPS = [1,2,4] # Predict 1 step (15 mins) ahead
NUM_PREDICTIONS = len(PREDICTION_STEPS)

IMG_SIZE = (540, 456)
PATCH_SIZE = (4, 8, 8)

OUTPUT_DIR = "/wetter/output"

BEST_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
LATEST_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "latest_checkpoint.pth")


RAIN_WEIGHT = 10.0



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    use_amp = device.type == 'cuda'

    print("Loading and splitting dataset...")
    full_dataset = WeatherDataset(
        root_dir=DATA_DIR,
        cache_dir=CACHE_DIR,
        window_size=WINDOW_SIZE,
        prediction=PREDICTION_STEPS,
        transform=None,
        cache_type='tensor',
        precision='uint8'
    )
    
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
  
    print("Initializing model...")
    base_model = SpatioTemporalTransformer(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        window_size=WINDOW_SIZE,
        in_chans=3, 
        embed_dim=96,
        depth=3,
        num_heads=4,
        num_predictions=NUM_PREDICTIONS
    ).to(device)
    
    base_model = torch.compile(base_model)

    criterion = WeightedBCELoss(rain_weight=RAIN_WEIGHT).to(device)
    optimizer = optim.AdamW(base_model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)

    scaler = torch.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    best_val_f1 = 0.0
    train_loss_history, val_loss_history = [], []
    train_f1_history, val_f1_history = [], []
    train_precision_history, val_precision_history = [], []
    train_recall_history, val_recall_history = [], []

    if os.path.exists(LATEST_CHECKPOINT_PATH):
        print(f"--- Resuming training from latest checkpoint: {LATEST_CHECKPOINT_PATH} ---")
        # Load checkpoint to CPU first to avoid GPU memory spike
        checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location='cpu')

        base_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint.get('epoch', 0) # Use .get for safety
        best_val_f1 = checkpoint.get('best_val_f1', 0.0)

        # Restore histories
        histories = checkpoint.get('histories', {})
        train_loss_history, val_loss_history = histories.get('loss', ([], []))
        train_f1_history, val_f1_history = histories.get('f1', ([], []))
        train_precision_history, val_precision_history = histories.get('precision', ([], []))
        train_recall_history, val_recall_history = histories.get('recall', ([], []))
        
        print(f"--- Resumed from epoch {start_epoch}. Best F1 so far: {best_val_f1:.4f} ---")
    else:
        print("--- No checkpoint found, starting training from scratch. ---")
    

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        base_model.train()
        running_train_loss = 0.0
        running_train_f1 = 0.0
        running_train_precision = 0.0
        running_train_recall = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets[:, :, 0:1, :, :]
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type="cuda",enabled=use_amp):
                outputs = base_model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_train_loss += loss.item()
           
            with torch.no_grad():
                metrics = calculate_binary_metrics(outputs, targets)
                running_train_f1 += metrics['f1']
                running_train_precision += metrics['precision']
                running_train_recall += metrics['recall']

            train_pbar.set_postfix(loss=loss.item(), f1=metrics['f1'], precision=metrics['precision'], recall=metrics['recall'])

        avg_train_loss = running_train_loss / len(train_loader)
        avg_train_f1 = running_train_f1 / len(train_loader)
        avg_train_precision = running_train_precision / len(train_loader)
        avg_train_recall = running_train_recall / len(train_loader)
        
        base_model.eval() 
        running_val_loss = 0.0
        running_val_f1 = 0.0
        running_val_precision = 0.0
        running_val_recall = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets[:, :, 0:1, :, :]                
                
                with torch.amp.autocast(device_type="cuda",enabled=use_amp):
                    outputs = base_model(inputs)
                    loss = criterion(outputs, targets)
                running_val_loss += loss.item()
                
                metrics = calculate_binary_metrics(outputs, targets)
                running_val_f1 += metrics['f1']
                running_val_precision += metrics['precision']
                running_val_recall += metrics['recall']
                val_pbar.set_postfix(loss=loss.item(), f1=metrics['f1'], precision=metrics['precision'], recall=metrics['recall'])

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_f1 = running_val_f1 / len(val_loader)
        avg_val_precision = running_val_precision / len(val_loader)
        avg_val_recall = running_val_recall / len(val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        print(f"  -> Train Metrics: F1={avg_train_f1:.4f}, Precision={avg_train_precision:.4f}, Recall={avg_train_recall:.4f}")
        print(f"  -> Val Metrics:   F1={avg_val_f1:.4f}, Precision={avg_val_precision:.4f}, Recall={avg_val_recall:.4f}")
        
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        train_f1_history.append(avg_train_f1)
        val_f1_history.append(avg_val_f1)
        train_precision_history.append(avg_train_precision)
        val_precision_history.append(avg_val_precision)
        train_recall_history.append(avg_train_recall)
        val_recall_history.append(avg_val_recall)

        latest_checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': base_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_f1': best_val_f1,
            'histories': {
                'loss': (train_loss_history, val_loss_history),
                'f1': (train_f1_history, val_f1_history),
                'precision': (train_precision_history, val_precision_history),
                'recall': (train_recall_history, val_recall_history),
            }
        }

        temp_latest_path = LATEST_CHECKPOINT_PATH + ".tmp"
        torch.save(latest_checkpoint_data, temp_latest_path)
        shutil.move(temp_latest_path, LATEST_CHECKPOINT_PATH)
        
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            print(f"  -> New best model found! Val F1: {best_val_f1:.4f}. Saving best checkpoint.")
            
            # Save the current state as the best checkpoint
            temp_best_path = BEST_CHECKPOINT_PATH + ".tmp"
            torch.save(latest_checkpoint_data, temp_best_path)
            shutil.move(temp_best_path, BEST_CHECKPOINT_PATH)

            # Also save a visual comparison for the best model
            save_validation_comparison(base_model, val_loader, device,
                                    save_path=os.path.join(OUTPUT_DIR, f"validation_comparison_{epoch}.png"))
        plot_history({
                        'loss': (train_loss_history, val_loss_history),
                        'f1': (train_f1_history, val_f1_history),
                        'precision': (train_precision_history, val_precision_history),
                        'recall': (train_recall_history, val_recall_history),
                    }, save_path=os.path.join(OUTPUT_DIR, "training_history.png"))

    print("\nTraining finished.")
    print(f"Best model saved at {BEST_CHECKPOINT_PATH} with a validation F1-score of {best_val_f1:.4f}")