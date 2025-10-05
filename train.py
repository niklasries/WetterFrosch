import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The video decoding*")

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader, random_split # type: ignore
from torchvision import transforms # type: ignore

import os
from tqdm import tqdm # type: ignore


from plot import plot_history, save_validation_comparison

from utils import WeatherDataset, SpecificValueWeightedMSELoss
from utils import calculate_f1_score
from model import SpatioTemporalTransformer

torch.set_float32_matmul_precision('high')

DATA_DIR  = "/wetter/input/WetterDaten/"
CACHE_DIR = "/wetter/input/WetterDatenCache/"

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 8
EPOCHS = 10

VALIDATION_SPLIT = 0.2 

WINDOW_SIZE = 16       # 12 steps = 3 hours
PREDICTION_STEPS = [1,2,4] # Predict 1 step (15 mins) ahead
NUM_PREDICTIONS = len(PREDICTION_STEPS)

IMG_SIZE = (530, 450)
PATCH_SIZE = (4, 10, 15)

OUTPUT_DIR = "/wetter/output"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

# Hex #6E6E6E is 110 in decimal. Normalized value is 110 / 255.
NO_RAIN_VALUE = 110.0 / 255.0
NO_RAIN_TOLERANCE = 0.02
RAIN_WEIGHT = 50.0

RAIN_THRESHOLD_F1 = NO_RAIN_VALUE + NO_RAIN_TOLERANCE


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
        transform=None
    )
    
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12,pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True, persistent_workers=True, prefetch_factor=2)
  
    print("Initializing model...")
    base_model = SpatioTemporalTransformer(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        window_size=WINDOW_SIZE,
        in_chans=3, 
        embed_dim=192,
        num_predictions=NUM_PREDICTIONS
    ).to(device)
    
    base_model = torch.compile(base_model)

    criterion = SpecificValueWeightedMSELoss(
        no_rain_value=NO_RAIN_VALUE,
        tolerance=NO_RAIN_TOLERANCE,
        rain_weight=RAIN_WEIGHT
    )
    optimizer = optim.AdamW(base_model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)

    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_mae = float('inf')
    best_val_f1 = 0.0
    train_loss_history, val_loss_history = [], []
    train_mae_history, val_mae_history = [], []
    train_f1_history, val_f1_history = [], []

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        base_model.train()
        running_train_loss = 0.0
        running_train_mae = 0.0
        running_train_f1 = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type="cuda",enabled=use_amp):
                outputs = base_model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_train_loss += loss.item()
           
            with torch.no_grad():
                mae = torch.mean(torch.abs(outputs - targets))
                running_train_mae += mae.item()
                f1 = calculate_f1_score(outputs, targets, RAIN_THRESHOLD_F1)
                running_train_f1 += f1

            train_pbar.set_postfix(loss=loss.item(), mae=mae.item(), f1=f1)

        avg_train_loss = running_train_loss / len(train_loader)
        avg_train_mae = running_train_mae / len(train_loader)
        avg_train_f1 = running_train_f1 / len(train_loader)
        
        base_model.eval() 
        running_val_loss = 0.0
        running_val_mae = 0.0
        running_val_f1 = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)                
                
                with torch.amp.autocast(device_type="cuda",enabled=use_amp):
                    outputs = base_model(inputs)
                    loss = criterion(outputs, targets)
                
                running_val_loss += loss.item()
                mae = torch.mean(torch.abs(outputs - targets))
                running_val_mae += mae.item()
                f1 = calculate_f1_score(outputs, targets, RAIN_THRESHOLD_F1)
                running_val_f1 += f1
                val_pbar.set_postfix(loss=loss.item(), mae=mae.item(), f1=f1)

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_mae = running_val_mae / len(val_loader)
        avg_val_f1 = running_val_f1 / len(val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Avg Train Loss: {avg_train_loss:.6f} | Avg Train MAE: {avg_train_mae:.6f} | Avg Train F1: {avg_train_f1:.4f} | "
              f"Avg Val Loss: {avg_val_loss:.6f} | Avg Val MAE: {avg_val_mae:.6f} | Avg Val F1: {avg_val_f1:.4f}")
        
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        train_mae_history.append(avg_train_mae)
        val_mae_history.append(avg_val_mae)
        train_f1_history.append(avg_train_f1)
        val_f1_history.append(avg_val_f1)
        
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            print(f"  -> New best model found! Saving model with Val F1: {best_val_f1:.4f}")
            torch.save(base_model.state_dict(), CHECKPOINT_PATH)
            save_validation_comparison(base_model, val_loader, device,
                                       save_path=os.path.join(OUTPUT_DIR, "validation_comparison.png"))
        plot_history(train_loss_history, val_loss_history, train_mae_history, val_mae_history, train_f1_history, val_f1_history,
                 save_path=os.path.join(OUTPUT_DIR, "training_history.png"))

    print("\nTraining finished.")
    print(f"Best model saved at {CHECKPOINT_PATH} with a validation MAE of {best_val_mae:.6f}")