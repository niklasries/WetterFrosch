# utils.py

import os
import re
from datetime import datetime
from typing import List
import multiprocessing
import json

import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.utils.data import Dataset # type: ignore
from torchvision.io import read_image, ImageReadMode, write_video, read_video # type: ignore
from tqdm import tqdm # type: ignore


def process_sample(args):
    
    start_index, all_timestamps, root_dir, cache_dir, window_size, prediction, crop_params = args
    
    y_start, x_start, crop_height, crop_width = crop_params
    
    start_ts = all_timestamps[start_index]
    timestamp_str = start_ts.strftime("%Y%m%d-%H%M")

    sat_video_path = os.path.join(cache_dir, f"sat_{timestamp_str}.mp4")
    radar_video_path = os.path.join(cache_dir, f"radar_{timestamp_str}.mp4")

    if os.path.exists(sat_video_path) and os.path.exists(radar_video_path):
        return True

    # 6E in hex is 110 in decimal. This is the raw pixel value.
    bg_color_uint8 = 110

   
    sat_frames = []
    for i in range(window_size):
        ts = all_timestamps[start_index + i]
        img_path = os.path.join(root_dir, f"sat_{ts.strftime('%Y%m%d-%H%M')}.png")
        tensor = read_image(img_path, mode=ImageReadMode.RGB)
        tensor = tensor[:, y_start:y_start+crop_height, x_start:x_start+crop_width]
        sat_frames.append(tensor)
    
    sat_video_tensor = torch.stack(sat_frames)
    sat_video_path = os.path.join(cache_dir, f"sat_{timestamp_str}.mp4")
    write_video(sat_video_path, sat_video_tensor.permute(0, 2, 3, 1), fps=4)

    radar_frames = []
    for step in prediction:
        target_ts_index = start_index + window_size + (step - 1)
        ts = all_timestamps[target_ts_index]
        img_path = os.path.join(root_dir, f"radar_{ts.strftime('%Y%m%d-%H%M')}.png")
        tensor = read_image(img_path, mode="RGB_ALPHA")
        rgb, alpha = tensor[0:3, :, :], tensor[3:4, :, :]

        background = torch.full_like(rgb, 0)
        rain = torch.full_like(rgb, 255)

        mask = (alpha == 0)
        mask_6e = (rgb == bg_color_uint8)

        filled_tensor = torch.where(mask, background, rain)
        filled_tensor = torch.where(mask_6e,background,filled_tensor)
        cropped_tensor = filled_tensor[:, y_start:y_start+crop_height, x_start:x_start+crop_width]
        radar_frames.append(cropped_tensor)
    
    radar_video_tensor = torch.stack(radar_frames)
    radar_video_path = os.path.join(cache_dir, f"radar_{timestamp_str}.mp4")
    write_video(radar_video_path, radar_video_tensor.permute(0, 2, 3, 1), fps=4)
    
    return True

class WeatherDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 cache_dir: str,
                 window_size: int = 12,
                 height: int = 540,
                 width: int = 456,
                 prediction: List[int] = [1], # [15,20,60] ->[1,2,4]
                 transform=None):

        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.transform = transform
        self.window_size = window_size
        self.prediction = sorted(prediction)
        self.max_step = self.prediction[-1]

        self._init_cache(height,width)
        self.video_files = sorted([f for f in os.listdir(self.cache_dir) if f.startswith('sat_') and f.endswith('.mp4')])

    def _init_cache(self,H,W):
        print("Verifying data cache...")
        os.makedirs(self.cache_dir, exist_ok=True)
        metadata_path = os.path.join(self.cache_dir, 'cache_info.json')

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)      
           
            if metadata.get('window_size') != self.window_size:
                raise ValueError(
                    f"FATAL: Cache was built with window_size={metadata.get('window_size')}, "
                    f"but current is {self.window_size}. Please delete the cache directory "
                    f"'{self.cache_dir}' and restart."
                )
            # Check for different prediction steps 
            if metadata.get('prediction_steps') != self.prediction:
                print("Prediction steps have changed. Invalidating and rebuilding radar video cache...")
                # Delete only the radar videos
                for filename in tqdm(os.listdir(self.cache_dir), desc="Deleting stale radar videos"):
                    if filename.startswith('radar_') and filename.endswith('.mp4'):
                        os.remove(os.path.join(self.cache_dir, filename))
                
        except (FileNotFoundError, json.JSONDecodeError):
            print("No valid cache metadata found. Cache will be built from scratch if needed.")
            # If no metadata, we assume cache is invalid
            metadata = {} 
        
        timestamp = self._get_timestamps(self.root_dir)
        start_indices = self._create_indices(timestamp)
        
        missing_indices = []
        for idx in start_indices:
            start_timestamp = timestamp[idx]
            timestamp_str = start_timestamp.strftime("%Y%m%d-%H%M")
            sat_video_path = os.path.join(self.cache_dir, f"sat_{timestamp_str}.mp4")
            radar_video_path = os.path.join(self.cache_dir, f"radar_{timestamp_str}.mp4")
            if not os.path.exists(sat_video_path) or not os.path.exists(radar_video_path):
                missing_indices.append(idx)
        
        if missing_indices:
            print(f"Cache is incomplete. Found {len(missing_indices)} missing samples to generate.")
            #crop params for Germany
            crop_params = (60, 220, H, W)
            
            tasks = [(idx, timestamp, self.root_dir, self.cache_dir, 
                      self.window_size, self.prediction, crop_params) for idx in missing_indices]
            num_workers = 8
            print(f"Starting processing with {num_workers} workers...")
            with multiprocessing.Pool(processes=num_workers) as pool:
                list(tqdm(pool.imap_unordered(process_sample, tasks), total=len(tasks), desc="Generating cached videos"))
            
            print("Cache generation complete. Writing metadata...")
            new_metadata = {'window_size': self.window_size, 'prediction_steps': self.prediction}
            with open(metadata_path, 'w') as f:
                json.dump(new_metadata, f, indent=4)
        else:
            print("Cache is complete and up-to-date.")

    def _get_timestamps(self, directory):
        timestamps = set()
        pattern = re.compile(r'\d{8}-\d{4}')
        for filename in os.listdir(directory):
            match = pattern.search(filename)
            if match: timestamps.add(match.group(0))
        return sorted([datetime.strptime(timestamp, "%Y%m%d-%H%M") for timestamp in timestamps])

    def _create_indices(self, timestamps):
        valid_indices = []
        num_timestamps = len(timestamps)
        last_idx = num_timestamps - self.window_size - self.max_step
        for i in range(last_idx + 1):
            valid_indices.append(i)
        return valid_indices

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        sat_video_filename = self.video_files[idx]
        radar_video_filename = sat_video_filename.replace('sat_', 'radar_')
        
        sat_video_path = os.path.join(self.cache_dir, sat_video_filename)
        radar_video_path = os.path.join(self.cache_dir, radar_video_filename)

        sat_frames, _, _ = read_video(sat_video_path, output_format="TCHW", pts_unit='sec')
        radar_frames, _, _ = read_video(radar_video_path, output_format="TCHW", pts_unit='sec')
        
        input_tensor = sat_frames.float() / 255.0
        target_tensor = radar_frames.float() / 255.0

        if self.transform:
            pass
            
        return input_tensor, target_tensor
    
class WeightedBCELoss(nn.Module):
    """
    A wrapper around PyTorch's BCEWithLogitsLoss that applies a weight to the
    positive class (rain) to handle class imbalance.
    """
    def __init__(self, rain_weight):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = torch.tensor([rain_weight])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, inputs, targets):
        return self.criterion(inputs, targets.float())

def calculate_binary_metrics(outputs, targets):
    """
    Calculates F1, precision, and recall for a binary classification task.
    """
    with torch.no_grad():
        # Apply sigmoid to convert logits to probabilities and threshold at 0.5 for binary classification
        preds_binary = (torch.sigmoid(outputs) > 0.5)
        targets_binary = targets.bool() # Ensure targets are boolean for bitwise ops
        
        # True Positives, False Positives, False Negatives
        tp = (preds_binary & targets_binary).sum().float()
        fp = (preds_binary & ~targets_binary).sum().float()
        fn = (~preds_binary & targets_binary).sum().float()
        
        # Precision and Recall
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        
        # F1-Score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        return {
            'f1': f1.item(),
            'precision': precision.item(),
            'recall': recall.item()
        }