"""
Clip-level collision detection dataset for the Collision Expert probe.

Samples random 16-frame windows from 128-frame videos.  50% of windows
contain at least one collision frame (label=1), 50% are collision-free (label=0).

Labels come from processed_proposals/sim_XXXXX.json → ground_truth.collisions.
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import json
import random
import numpy as np


class CLEVRERCollisionDataset(Dataset):
    """Sample 16-frame windows + binary collision label."""

    def __init__(self, split='train', num_frames=16, frame_size=96,
                 tensor_dir=None, ann_dir=None):
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.tensor_dir = tensor_dir
        self.ann_dir = ann_dir
        self.total_frames = 128  # CLEVRER videos

        self.resize = T.Resize((frame_size, frame_size), antialias=True)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        )

        # collision frames per video: {video_idx: [frame, frame, ...]}
        self._collision = {}
        self._video_cache = {}

    def _get_collisions(self, video_idx):
        if video_idx in self._collision:
            return self._collision[video_idx]
        ann_path = os.path.join(self.ann_dir, f"sim_{video_idx:05d}.json")
        frames = []
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                data = json.load(f)
            for c in data.get('ground_truth', {}).get('collisions', []):
                frames.append(c['frame'])
        self._collision[video_idx] = frames
        return frames

    def _load_video(self, video_idx):
        if video_idx in self._video_cache:
            return self._video_cache[video_idx]
        path = os.path.join(self.tensor_dir, f"video_{video_idx:05d}.pth")
        if os.path.exists(path):
            v = torch.load(path, map_location='cpu', weights_only=False).float() / 255.0
        else:
            v = torch.zeros(3, self.total_frames, self.frame_size, self.frame_size)
        self._video_cache[video_idx] = v
        return v

    def __len__(self):
        return 8000 if self.split == 'train' else 2000

    def __getitem__(self, idx):
        if self.split == 'validation':
            idx = idx + 8000

        video = self._load_video(idx)              # (3, 128, H, W)
        collisions = self._get_collisions(idx)      # [frame_id, ...]

        # 50% collision clip, 50% non-collision
        if random.random() < 0.5 and collisions:
            # pick a random collision frame, centre window around it
            cf = random.choice(collisions)
            start = cf - self.num_frames // 2
            start = max(0, min(start, self.total_frames - self.num_frames))
            start = random.randint(
                max(0, cf - self.num_frames),
                min(cf, self.total_frames - self.num_frames)
            )
            label = 1.0
        else:
            # sample a collision-free window
            collision_set = set(collisions)
            valid_starts = [
                s for s in range(0, self.total_frames - self.num_frames + 1)
                if not collision_set.intersection(range(s, s + self.num_frames))
            ]
            if valid_starts:
                start = random.choice(valid_starts)
            else:
                start = 0  # fallback (shouldn't happen with real CLEVRER)
            label = 0.0

        indices = list(range(start, start + self.num_frames))
        clip = video[:, indices]                    # (3, 16, H, W)
        if clip.shape[-1] != self.frame_size:
            clip = self.resize(clip)
        processed = torch.stack([self.normalize(clip[:, t]) for t in range(self.num_frames)], dim=1)
        return processed, torch.tensor(label, dtype=torch.float32)
