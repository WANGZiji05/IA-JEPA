"""
Evaluate Collision Expert Probe on frozen JEPA backbones.

Usage:
    python src/eval/evaluate_collision.py \
        --variant physics_aware \
        --checkpoint checkpoints/physics_aware/last.pth \
        --batch_size 256 \
        --tensor_dir /path/to/tensors \
        --ann_dir /path/to/processed_proposals
"""

import torch
from torch.utils.data import DataLoader
import argparse
import os

from src.data.clevrer_collision_dataset import CLEVRERCollisionDataset
from src.eval.collision_probe import train_collision_probe, evaluate_collision_probe
from src.models.jepa_baseline import VideoJEPA
from src.models.jepa_object import ObjectMaskedJEPA
from src.models.jepa_interaction import InteractionAwareJEPA
from src.models.jepa_physics_aware import PhysicsAwareJEPA
from src.models.jepa_random_tube import RandomTubeJEPA
from src.models.jepa_multiblock import MultiBlockJEPA
from src.models.jepa_pa_prob import PAProbabilisticJEPA
from src.models.jepa_pa_block import PABlockJEPA
from src.models.jepa_mixed_pa import MixedPhysicsAwareJEPA


def collate(batch):
    videos = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return videos, labels


def evaluate(variant, checkpoint_path, device, batch_size, tensor_dir, ann_dir):
    print(f"Collision Expert Probe — {variant}")

    # Load model
    params = {"img_size": 96, "num_frames": 16}
    if variant == 'baseline':
        model = VideoJEPA(**params)
    elif variant == 'object':
        model = ObjectMaskedJEPA(**params)
    elif variant == 'interaction':
        model = InteractionAwareJEPA(**params)
    elif variant == 'physics_aware':
        model = PhysicsAwareJEPA(**params)
    elif variant == 'mixed_pa':
        from src.models.jepa_mixed_pa import MixedPhysicsAwareJEPA
        model = MixedPhysicsAwareJEPA(**params)
    elif variant == 'random_tube':
        model = RandomTubeJEPA(**params)
    elif variant == 'multiblock':
        model = MultiBlockJEPA(**params)
    elif variant == 'pa_prob':
        model = PAProbabilisticJEPA(**params)
    elif variant == 'pa_block':
        model = PABlockJEPA(**params)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Data
    train_ds = CLEVRERCollisionDataset(
        split='train', tensor_dir=tensor_dir, ann_dir=ann_dir,
    )
    val_ds = CLEVRERCollisionDataset(
        split='validation', tensor_dir=tensor_dir, ann_dir=ann_dir,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)

    # Train probe
    probe = train_collision_probe(model, train_loader, val_loader, device, epochs=10)

    # Final evaluation
    train_acc = evaluate_collision_probe(probe, model, train_loader, device)
    val_acc = evaluate_collision_probe(probe, model, val_loader, device)
    print(f"\n=== Collision Expert Results [{variant}] ===")
    print(f"  Train accuracy:  {train_acc:.4f}")
    print(f"  Val accuracy:    {val_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--tensor_dir', required=True)
    parser.add_argument('--ann_dir', required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    evaluate(args.variant, args.checkpoint, args.device, args.batch_size,
             args.tensor_dir, args.ann_dir)
