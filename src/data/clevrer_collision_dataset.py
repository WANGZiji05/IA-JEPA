"""
Binary collision detection dataset for the Collision Expert probe.

Each video clip is labelled 1 if it contains at least one collision event,
0 otherwise. Labels come from processed_proposals/sim_XXXXX.json →
ground_truth.collisions.
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import json


class CLEVRERCollisionDataset(Dataset):
    """Load video tensors + binary collision labels."""

    def __init__(self, split='train', num_frames=16, frame_size=96,
                 tensor_dir=None, ann_dir=None):
        self.num_videos = 10000
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.tensor_dir = tensor_dir
        self.ann_dir = ann_dir

        self.resize = T.Resize((frame_size, frame_size), antialias=True)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        )

        # load collision labels: video_idx → has_collision (0/1)
        self.labels = self._load_labels()

    def _load_labels(self):
        labels = {}
        for idx in range(self.num_videos):
            ann_path = os.path.join(self.ann_dir, f"sim_{idx:05d}.json")
            has_collision = 0
            if os.path.exists(ann_path):
                with open(ann_path, 'r') as f:
                    data = json.load(f)
                collisions = data.get('ground_truth', {}).get('collisions', [])
                if len(collisions) > 0:
                    has_collision = 1
            labels[idx] = has_collision
        return labels

    def __len__(self):
        # 0-7999 train, 8000-9999 val
        return 8000 if self.split == 'train' else 2000

    def __getitem__(self, idx):
        if self.split == 'validation':
            idx = idx + 8000

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # load video
        fast_path = os.path.join(self.tensor_dir, f"video_{idx:05d}.pth")
        if os.path.exists(fast_path):
            full_video = torch.load(fast_path, map_location='cpu', weights_only=False)
            total_frames = full_video.shape[1]
            indices = torch.linspace(0, total_frames - 1, steps=self.num_frames).long().tolist()
            clip = full_video[:, indices].float() / 255.0
            if clip.shape[-1] != self.frame_size:
                clip = self.resize(clip)
        else:
            clip = torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)

        processed = [self.normalize(clip[:, t]) for t in range(self.num_frames)]
        clip = torch.stack(processed, dim=1)
        return clip, label
