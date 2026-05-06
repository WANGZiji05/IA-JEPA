import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import torchvision.transforms as T
import numpy as np
import os
import json
try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None

class CLEVRERVideoDataset(Dataset):
    """
    A PyTorch Dataset for loading CLEVRER videos from preprocessed tensors.
    """
    def __init__(self, split='train', num_frames=16, frame_size=112, 
                 mask_dir=None, ann_dir=None, tensor_dir=None):
        super().__init__()
        # We don't load the HF dataset here to avoid the label issue
        # We just use the number of videos we expect
        self.num_videos = 10000 if split == 'train' else 5000
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.mask_dir = mask_dir
        self.ann_dir = ann_dir
        self.tensor_dir = tensor_dir
        
        self.resize = T.Resize((frame_size, frame_size), antialias=True)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])

    def __len__(self):
        return self.num_videos

    def _get_masks(self, video_idx, frame_indices):
        if self.mask_dir is None or mask_utils is None:
            return None
        mask_path = os.path.join(self.mask_dir, f"proposal_{video_idx:05d}.json")
        if not os.path.exists(mask_path):
            return None
        with open(mask_path, 'r') as f:
            data = json.load(f)
        all_masks = []
        for f_idx in frame_indices:
            if f_idx >= len(data['frames']):
                all_masks.append(torch.zeros(1, self.frame_size, self.frame_size))
                continue
            frame_data = data['frames'][f_idx]
            if not frame_data['objects']:
                all_masks.append(torch.zeros(1, self.frame_size, self.frame_size))
                continue
            h, w = frame_data['objects'][0]['mask']['size']
            combined_mask = np.zeros((h, w), dtype=np.float32)
            for obj in frame_data['objects']:
                m = mask_utils.decode(obj['mask'])
                combined_mask = np.maximum(combined_mask, m.astype(np.float32))
            mask_tensor = torch.from_numpy(combined_mask).unsqueeze(0)
            mask_tensor = T.functional.resize(mask_tensor, (self.frame_size, self.frame_size), 
                                              interpolation=T.InterpolationMode.NEAREST)
            all_masks.append(mask_tensor)
        return torch.stack(all_masks, dim=1) # [1, T, H, W]

    def _get_collisions(self, video_idx):
        if self.ann_dir is None:
            return None
        chunk_start = (video_idx // 1000) * 1000
        chunk_dir = os.path.join(self.ann_dir, f"annotation_{chunk_start:05d}-{chunk_start+1000:05d}")
        ann_path = os.path.join(chunk_dir, f"annotation_{video_idx:05d}.json")
        
        if not os.path.exists(ann_path):
            return None
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        collision_frames = [c['frame_id'] for c in data.get('collision', [])]
        if not collision_frames:
            return torch.tensor([], dtype=torch.long)
        return torch.tensor(collision_frames, dtype=torch.long)

    def __getitem__(self, idx):
        # Apply offset for validation/test splits
        real_idx = idx
        if self.split == 'validation':
            real_idx = idx + 10000
        elif self.split == 'test':
            real_idx = idx + 15000
            
        # 1. Try Fast Path (Pre-processed Tensors)
        fast_path = os.path.join(self.tensor_dir, f"video_{real_idx:05d}.pth") if self.tensor_dir else None
        
        if fast_path and os.path.exists(fast_path):
            try:
                # Load raw uint8 tensor [C, T_total, H, W]
                full_video = torch.load(fast_path, map_location='cpu')
                total_frames = full_video.shape[1]
                
                # Temporal Striding: Sample num_frames uniformly across the entire video
                # This ensures consistent evaluation for rollout stability
                indices = torch.linspace(0, total_frames - 1, steps=self.num_frames).long().tolist()
                
                clip = full_video[:, indices].float() / 255.0
                
                # Resize if needed
                if clip.shape[-1] != self.frame_size:
                    clip = self.resize(clip)
            except Exception:
                # Fallback to zeros if loading fails
                clip = torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
                indices = [0] * self.num_frames
        else:
            # Fallback to zeros if tensor not found
            clip = torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
            indices = [0] * self.num_frames

        # 2. Common Post-processing
        processed_frames = [self.normalize(clip[:, t]) for t in range(self.num_frames)]
        clip_final = torch.stack(processed_frames, dim=1) 
        
        masks = self._get_masks(idx, indices)
        if masks is None:
            masks = torch.zeros(1, self.num_frames, self.frame_size, self.frame_size)
            
        collisions = self._get_collisions(idx)
        if collisions is None:
            # Return a fixed-size tensor for collisions to allow batching
            collisions = torch.full((10,), -1, dtype=torch.long)
        else:
            # Pad or truncate collisions to fixed size 10
            if collisions.shape[0] < 10:
                collisions = torch.cat([collisions, torch.full((10 - collisions.shape[0],), -1, dtype=torch.long)])
            else:
                collisions = collisions[:10]
        
        res = {
            "video": clip_final,
            "masks": masks,
            "collisions": collisions
        }
        return res, 0
