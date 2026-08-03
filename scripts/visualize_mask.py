"""
Visualize masking strategies — save PNGs for offline viewing.

Usage on server:
    python scripts/visualize_mask.py --variant baseline --checkpoint checkpoints/last.pth

Output: checkpoints/<variant>/mask_viz/  — PNG files, one per sample.
scp them to local machine and open.
"""

import argparse, os, torch, numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

from src.models.jepa_baseline import VideoJEPA
from src.models.jepa_object import ObjectMaskedJEPA
from src.models.jepa_interaction import InteractionAwareJEPA
from src.models.jepa_physics_aware import PhysicsAwareJEPA
from src.models.jepa_mixed_pa import MixedPhysicsAwareJEPA


def load_model(variant, checkpoint_path, device):
    params = {"img_size": 96, "num_frames": 16, "mask_ratio": 0.6}
    cls = {
        'baseline': VideoJEPA, 'object': ObjectMaskedJEPA,
        'interaction': InteractionAwareJEPA, 'physics_aware': PhysicsAwareJEPA,
        'mixed_pa': MixedPhysicsAwareJEPA,
    }[variant]
    model = cls(**params).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def get_mask(model, variant, video_tensor, device):
    """Returns context_idx, target_idx for the given video batch."""
    v = video_tensor.unsqueeze(0).to(device)  # (1, C, T, H, W)

    if variant == 'baseline':
        B, N = 1, (v.shape[2] // 2) * (v.shape[3] // 16) * (v.shape[4] // 16)
        ids_keep, ids_mask = model.generate_mask(B, N, device)
        return ids_keep, ids_mask

    # For PA / Mixed-PA / Interaction: call get_physics_mask or get_interaction_mask
    if hasattr(model, 'get_physics_mask'):
        ctx, tgt = model.get_physics_mask(v)
        return ctx, tgt
    if hasattr(model, 'get_interaction_mask'):
        ctx, tgt = model.get_interaction_mask(v)
        return ctx, tgt
    # Object
    if hasattr(model, 'get_mask_indices'):
        ctx, tgt = model.get_mask_indices(v)
        Td, Sd = 8, 36  # Td=16//2, Sd=(96//16)^2
        ctx = torch.cat([ctx + t * Sd for t in range(Td)], dim=1)
        tgt = torch.cat([tgt + t * Sd for t in range(Td)], dim=1)
        return ctx, tgt

    raise RuntimeError("Unknown masking method")


def visualize(variant, checkpoint_path, tensor_dir, out_dir, num_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(variant, checkpoint_path, device)
    os.makedirs(out_dir, exist_ok=True)

    # Find available videos
    vids = [f for f in sorted(os.listdir(tensor_dir)) if f.endswith('.pth')][:num_samples + 5]
    # Use videos with collisions if possible
    import json
    ann_dir = '/research/d7/spc/yrwang5/JEPA_experiements/data/clevrer/processed_proposals'
    collision_ids = []
    for i in range(100):
        path = os.path.join(ann_dir, f'sim_{i:05d}.json')
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if len(data.get('ground_truth', {}).get('collisions', [])) > 0:
                collision_ids.append(i)
                if len(collision_ids) >= num_samples:
                    break

    Td, Hp, Wp = 8, 6, 6  # tubelet grid dimensions

    for i, vid_idx in enumerate(collision_ids):
        path = os.path.join(tensor_dir, f'video_{vid_idx:05d}.pth')
        if not os.path.exists(path):
            continue
        video = torch.load(path, map_location='cpu', weights_only=False).float()
        total_frames = video.shape[1]
        indices = torch.linspace(0, total_frames - 1, steps=16).long()
        clip = video[:, indices] / 255.0

        # Get mask
        with torch.no_grad():
            ctx, tgt = get_mask(model, variant, clip, device)
        ctx = ctx.cpu().squeeze(0).numpy()
        tgt = tgt.cpu().squeeze(0).numpy()

        # Build mask overlaid on a representative frame (frame 8, middle of 16)
        frame = clip[:, 8].permute(1, 2, 0).numpy()  # (H, W, C)
        frame = np.clip(frame, 0, 1)

        # Create mask overlay for this frame's spatial patches
        t_frame = 8 // 2  # tubelet temporal index
        mask_vis = np.ones((Hp, Wp, 3)) * 0.3  # grey = neither
        for p in ctx:
            t_idx, s_idx = divmod(p, Hp * Wp)
            if t_idx == t_frame:
                r, c = divmod(s_idx, Wp)
                mask_vis[r, c] = [0.2, 0.8, 0.2]  # green = context
        for p in tgt:
            t_idx, s_idx = divmod(p, Hp * Wp)
            if t_idx == t_frame:
                r, c = divmod(s_idx, Wp)
                mask_vis[r, c] = [0.9, 0.2, 0.2]  # red = target

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(frame)
        ax1.set_title(f'Frame (video {vid_idx})')
        ax1.axis('off')

        ax2.imshow(mask_vis, interpolation='nearest')
        ax2.set_title(f'{variant}\ngreen=context(visible)  red=target(predict)')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{variant}_{vid_idx:05d}.png'), dpi=150)
        plt.close()
        print(f'Saved {variant}_{vid_idx:05d}.png')

    print(f'Done. View in {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tensor_dir', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--num', type=int, default=5)
    args = parser.parse_args()

    if args.tensor_dir is None:
        args.tensor_dir = '/research/d7/spc/yrwang5/JEPA_experiements/data/clevrer/tensors'
    if args.out is None:
        args.out = os.path.join(os.path.dirname(args.checkpoint), 'mask_viz')

    visualize(args.variant, args.checkpoint, args.tensor_dir, args.out, args.num)
