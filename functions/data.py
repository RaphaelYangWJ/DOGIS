import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# Flow Matching Dataset --> Inverse
# === Class: Flow Matching Dataset (Sobolev & Random Mask Edition) ===
class FM_Dataset(Dataset):
    def __init__(self, data_dir, num_sensor_points=15):
        """
        num_sensor_points: The number of observation points in each random sampling (e.g., 15 points for extreme sparsity).
        """
        # Read dataset
        with h5py.File(data_dir, 'r') as f:
            self.target = f['target'][:]
            self.dense_obs = f['dense_obs'][:]
            self.target_scaler_params = f['target_scaler_params'][:]
            self.obs_scaler_params = f['obs_scaler_params'][:]
   
        # Tensor conversion
        self.target = torch.tensor(self.target, dtype=torch.float32)
        self.dense_obs = torch.tensor(self.dense_obs, dtype=torch.float32)

        # Record the number of random points
        self.num_sensor_points = num_sensor_points
        
        # Extract basic dimensional information (used for subsequent acquisition of network parameters).
        self.field_channels = self.target.shape[1]
        self.obs_channels = self.dense_obs.shape[1]
        self.field_size = self.target.shape[-1]
        
        # Global vector length = Number of observed variable channels * Number of observation points
        self.global_feat_size = self.obs_channels * self.num_sensor_points

        print(f"Dataset Loaded. Sobolev Gradients & Random {self.num_sensor_points}-Point Masking Enabled.")


    def _compute_spatial_gradients(self, tensor):
        """
        Compute second-order central difference using PyTorch's native gradient operators (edges are forward/backward differences)
        Input: [C, H, W]
        Output: [2*C, H, W] (the first part is dx, the second part is dy)
        """
        # dim=(-2, -1) represents taking the derivative with respect to the last two dimensions (y-axis, x-axis).
        grad_y, grad_x = torch.gradient(tensor, dim=(-2, -1))
        
        # Concatenate along the channel dimension; if the input has 21 channels, the output will have 42 channels.
        return torch.cat([grad_x, grad_y], dim=0)


    def _create_random_mask(self, size, num_points):
        """
        On a full graph of size * size, randomly sprinkle num_points points.
        """
        mask = torch.zeros((size, size), dtype=torch.float32)
        
        # Generate random, non-repeating indices from 0 to size*size-1
        random_indices = torch.randperm(size * size)[:num_points]
        
        # Flatten the mask, enter 1, then transform it back to 2D
        mask.view(-1)[random_indices] = 1.0
        return mask

    def get_size_params(self):
        return {
            "field_channels": self.field_channels,
            "field_size": self.field_size,
            "obs_channel": self.obs_channels,
            "global_feat_size": self.global_feat_size,
            "target_scaler_params": self.target_scaler_params,
            "obs_scaler_params": self.obs_scaler_params,
        }

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        # 1. Obtaining clean, realistic physics
        target_img = self.target[idx]       # [C_t, 64, 64]
        dense_obs_img = self.dense_obs[idx] # [C_o, 64, 64]

        # 2. Generate a random mask in real time for the current sample.
        mask = self._create_random_mask(self.field_size, self.num_sensor_points) # [64, 64]

        # 3Applying a mask results in sparse observations.
        sparse_obs = dense_obs_img * mask.unsqueeze(0) # [C_o, 64, 64]

        # 4. Generate Global Feature (extracting valid points in left-to-right, top-to-bottom order)
        mask_bool = mask > 0.5
        selected_points = dense_obs_img[:, mask_bool] # [C_o, num_points]
        global_feat = selected_points.flatten()       # [C_o * num_points]
 
        if sparse_obs.shape[0] == 5:
            target_len = 80
        else:
            target_len = 16

        current_len = global_feat.shape[0]

        if current_len > target_len:
            global_feat = global_feat[:target_len]
        elif current_len < target_len:
            global_feat = F.pad(global_feat, (0, target_len - current_len), mode='constant', value=0.0)

        return {
            "target": target_img,
            "dense_obs": dense_obs_img,
            "spatial_feat": sparse_obs,
            "global_feat": global_feat,
            "mask": mask       
        }



# FNO Training Dataset --> Forward
class FNO_Dataset(Dataset):
    def __init__(self, data_dir):
        # Read dataset
        with h5py.File(data_dir, 'r') as f:
            self.target = f['target'][:]
            self.dense_obs = f['dense_obs'][:]
            self.target_scaler_params = f['target_scaler_params'][:]
            self.obs_scaler_params = f['obs_scaler_params'][:]
        # Tensor conversion
        self.target = torch.tensor(self.target, dtype=torch.float32)
        self.dense_obs = torch.tensor(self.dense_obs, dtype=torch.float32)
        print(f"Dataset Loaded.")

    def get_size_params(self):
        return {
            "field_channels": self.target.shape[1],
            "field_size": self.target.shape[-1],
            "obs_channels": self.dense_obs.shape[1],
            "obs_size": self.dense_obs.shape[-1],}

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        return {"input_field": self.target[idx],
                "label_obs": self.dense_obs[idx],}




# === Func: Flow Matching Dataloader
def FM_dataloader(data_dir, batch_size=32, num_sensor_points=25, shuffle=True, num_workers=4,):
    dataset = FM_Dataset(data_dir, num_sensor_points)
    model_params = dataset.get_size_params()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader, model_params


# === Func: FNO Dataloader
def fno_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=4,):
    dataset = FNO_Dataset(data_dir)
    model_params = dataset.get_size_params()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader, model_params