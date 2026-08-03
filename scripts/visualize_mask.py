"""
Visualize masking around collision frames — continuous sequence with overlay.

Saves a multi-panel PNG per variant showing 5 consecutive frames centered on a
collision event, with mask overlaid semi-transparent on the original frames.

Usage on server:
    python scripts/visualize_mask.py --variant baseline --checkpoint checkpoints/last.pth
"""

import argparse, os, torch, numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.models.jepa_baseline import VideoJEPA
from src.models.jepa_object import ObjectMaskedJEPA
from src.models.jepa_interaction import InteractionAwareJEPA
from src.models.jepa_physics_aware import PhysicsAwareJEPA
from src.models.jepa_mixed_pa import MixedPhysicsAwareJEPA


def load_model(variant, checkpoint_path, device):
    params = {"img_size": 96, "num_frames": 16, "mask_ratio": 0.6}
    cls = {'baseline': VideoJEPA, 'object': ObjectMaskedJEPA,
           'interaction': InteractionAwareJEPA, 'physics_aware': PhysicsAwareJEPA,
           'mixed_pa': MixedPhysicsAwareJEPA}[variant]
    model = cls(**params).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def find_collision_videos(ann_dir, num_videos=5):
    """Find videos with collisions and return (vid_idx, collision_frame)."""
    results = []
    for idx in range(500):
        path = os.path.join(ann_dir, f'sim_{idx:05d}.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        collisions = data.get('ground_truth', {}).get('collisions', [])
        if collisions:
            cf = collisions[len(collisions) // 2]['frame']  # middle collision
            results.append((idx, cf))
            if len(results) >= num_videos:
                break
    return results


def get_mask(model, variant, clip, device):
    """Get context and target indices for a 16-frame clip."""
    v = clip.unsqueeze(0).to(device)

    if variant == 'baseline':
        Td, Hp, Wp = 16 // 2, 96 // 16, 96 // 16
        N = Td * Hp * Wp
        ids_keep, ids_mask = model.generate_mask(1, N, device)
        return ids_keep, ids_mask

    if hasattr(model, 'get_physics_mask'):
        return model.get_physics_mask(v)
    if hasattr(model, 'get_interaction_mask'):
        return model.get_interaction_mask(v)
    if hasattr(model, 'get_mask_indices'):
        ctx, tgt = model.get_mask_indices(v)
        Td, Sd = 16 // 2, (96 // 16) ** 2
        ctx_full = torch.cat([ctx + t * Sd for t in range(Td)], dim=1)
        tgt_full = torch.cat([tgt + t * Sd for t in range(Td)], dim=1)
        return ctx_full, tgt_full
    raise RuntimeError("Unknown masking method")


def visualize_one(video_path, cf, model, variant, device, out_path, tubelet_size=2):
    """Draw 5-frame strip centered on collision, with mask overlay."""
    video = torch.load(video_path, map_location='cpu', weights_only=False).float()

    # Sample 16 frames with collision frame near the middle
    total_frames = video.shape[1]
    cf_idx = cf  # original frame index in [0, 127]

    # Pick a 16-frame window that includes the collision
    start = max(0, min(cf_idx - 8, total_frames - 16))
    indices = list(range(start, start + 16))
    clip = video[:, indices] / 255.0  # (3, 16, 96, 96)

    # Get mask
    with torch.no_grad():
        ctx, tgt = get_mask(model, variant, clip, device)
    ctx = ctx.cpu().squeeze(0).numpy()
    tgt = tgt.cpu().squeeze(0).numpy()

    Hp, Wp = 6, 6
    patch_size = 16
    Td = 8  # 16 // tubelet_size

    # Find which frame indices to show (around collision)
    cf_local = cf_idx - start  # collision frame in the 16-frame clip
    show_frames = [max(0, cf_local - 2), max(0, cf_local - 1),
                   cf_local, min(15, cf_local + 1), min(15, cf_local + 2)]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

    for ax_idx, f_idx in enumerate(show_frames):
        ax = axes[ax_idx]
        frame = clip[:, f_idx].permute(1, 2, 0).numpy()
        frame = np.clip(frame, 0, 1)

        ax.imshow(frame)

        # Overlay mask for this frame's spatial patches
        t_tube = f_idx // tubelet_size  # which tubelet temporal index
        overlay = np.zeros((96, 96, 4))

        for p in ctx:
            t_idx, s_idx = divmod(p, Hp * Wp)
            if t_idx == t_tube:
                r, c = divmod(s_idx, Wp)
                y0, x0 = r * patch_size, c * patch_size
                overlay[y0:y0+patch_size, x0:x0+patch_size] = [0.2, 0.8, 0.2, 0.45]  # green

        for p in tgt:
            t_idx, s_idx = divmod(p, Hp * Wp)
            if t_idx == t_tube:
                r, c = divmod(s_idx, Wp)
                y0, x0 = r * patch_size, c * patch_size
                overlay[y0:y0+patch_size, x0:x0+patch_size] = [0.9, 0.2, 0.2, 0.45]  # red

        ax.imshow(overlay)

        # Label collision frame
        label = f"Frame {f_idx + start}" + (" ★COLLISION" if f_idx == cf_local else "")
        ax.set_title(label, fontsize=10)
        ax.axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0.2, 0.8, 0.2, 0.6), label='Context (visible)'),
        Patch(facecolor=(0.9, 0.2, 0.2, 0.6), label='Target (predict)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11)
    plt.suptitle(f'{variant.upper()} — collision at frame {cf_idx}', fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tensor_dir', default=None)
    parser.add_argument('--ann_dir', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--num', type=int, default=3)
    args = parser.parse_args()

    if args.tensor_dir is None:
        args.tensor_dir = '/research/d7/spc/yrwang5/JEPA_experiements/data/clevrer/tensors'
    if args.ann_dir is None:
        args.ann_dir = '/research/d7/spc/yrwang5/JEPA_experiements/data/clevrer/processed_proposals'
    if args.out is None:
        args.out = os.path.join(os.path.dirname(args.checkpoint), 'mask_viz')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.variant, args.checkpoint, device)
    os.makedirs(args.out, exist_ok=True)

    videos = find_collision_videos(args.ann_dir, args.num)
    for vid_idx, cf in videos:
        vpath = os.path.join(args.tensor_dir, f'video_{vid_idx:05d}.pth')
        out_path = os.path.join(args.out, f'{args.variant}_{vid_idx:05d}_cf{cf}.png')
        visualize_one(vpath, cf, model, args.variant, device, out_path)
        print(f'Saved {out_path}')

    print(f'Done. {len(videos)} visualizations in {args.out}')


if __name__ == '__main__':
    main()
