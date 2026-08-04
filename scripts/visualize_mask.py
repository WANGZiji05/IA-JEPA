"""
Visualize masking strategies — 5-frame sequence around collision, blue overlay.

Output: <variant>_mask_visualization_01.png ... _03.png
"""

import argparse, os, torch, numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.jepa_baseline import VideoJEPA
from src.models.jepa_object import ObjectMaskedJEPA
from src.models.jepa_interaction import InteractionAwareJEPA
from src.models.jepa_physics_aware import PhysicsAwareJEPA
from src.models.jepa_mixed_pa import MixedPhysicsAwareJEPA
from src.models.jepa_random_tube import RandomTubeJEPA
from src.models.jepa_multiblock import MultiBlockJEPA
from src.models.jepa_pa_prob import PAProbabilisticJEPA
from src.models.jepa_pa_block import PABlockJEPA


MODEL_MAP = {
    'baseline': VideoJEPA, 'object': ObjectMaskedJEPA,
    'interaction': InteractionAwareJEPA, 'physics_aware': PhysicsAwareJEPA,
    'mixed_pa': MixedPhysicsAwareJEPA, 'random_tube': RandomTubeJEPA,
    'multiblock': MultiBlockJEPA, 'pa_prob': PAProbabilisticJEPA,
    'pa_block': PABlockJEPA,
}


def find_collision_videos(ann_dir, num=3):
    results = []
    for idx in range(500):
        path = os.path.join(ann_dir, f'sim_{idx:05d}.json')
        if not os.path.exists(path): continue
        with open(path) as f:
            data = json.load(f)
        collisions = data.get('ground_truth', {}).get('collisions', [])
        if collisions:
            cf = collisions[len(collisions) // 2]['frame']
            results.append((idx, cf))
            if len(results) >= num: break
    return results


def get_mask(model, variant, clip, device):
    v = clip.unsqueeze(0).to(device)
    if variant == 'baseline':
        Td, Hp, Wp = 8, 6, 6
        N = Td * Hp * Wp
        ids_keep, ids_mask = model.generate_mask(1, N, device)
        return ids_keep, ids_mask
    if hasattr(model, 'get_physics_mask'):
        return model.get_physics_mask(v)
    if hasattr(model, 'get_interaction_mask'):
        return model.get_interaction_mask(v)
    if hasattr(model, 'get_mask_indices'):
        dummy = torch.zeros(1, 1, 16, 96, 96, device=device)
        ctx, tgt = model.get_mask_indices(v, masks=dummy)
        Td, Sd = 8, 36
        ctx = torch.cat([ctx + t * Sd for t in range(Td)], dim=1)
        tgt = torch.cat([tgt + t * Sd for t in range(Td)], dim=1)
        return ctx, tgt
    if hasattr(model, 'generate_blocks'):
        model._Td, model._Hp, model._Wp = 8, 6, 6
        return model.generate_blocks(1, device)
    if hasattr(model, 'generate_mask'):
        return model.generate_mask(1, 288, device)
    raise RuntimeError(f'Unknown mask method for {variant}')


def visualize_one(variant, video_path, cf, model, device, out_path):
    video = torch.load(video_path, map_location='cpu', weights_only=False).float()
    total_frames = video.shape[1]
    cf_local = cf

    # 16-frame window containing collision near middle
    start = max(0, min(cf_local - 8, total_frames - 16))
    indices = list(range(start, start + 16))
    clip = video[:, indices] / 255.0
    cf_in_clip = cf_local - start

    with torch.no_grad():
        ctx, tgt = get_mask(model, variant, clip, device)
    ctx = ctx.cpu().squeeze(0).numpy()
    tgt = tgt.cpu().squeeze(0).numpy()

    Hp, Wp, ps = 6, 6, 16
    Td = 8

    # 5 frames with gap: c-10, c-5, c, c+5, c+10 (in original frame space)
    show_frames_orig = [cf - 10, cf - 5, cf, cf + 5, cf + 10]
    show_frames_orig = [max(0, min(127, f)) for f in show_frames_orig]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))
    for ax_idx, f_orig in enumerate(show_frames_orig):
        ax = axes[ax_idx]
        f_local = f_orig - start
        if f_local < 0 or f_local >= 16:
            ax.axis('off'); continue

        frame = clip[:, f_local].permute(1, 2, 0).numpy()
        frame = np.clip(frame, 0, 1)
        ax.imshow(frame)

        t_tube = f_local // 2
        overlay = np.zeros((96, 96, 4))
        for p in tgt:
            t_idx, s_idx = divmod(p, 36)
            if t_idx == t_tube:
                r, c = divmod(s_idx, 6)
                overlay[r*ps:(r+1)*ps, c*ps:(c+1)*ps] = [0.1, 0.4, 0.9, 0.55]  # blue semi-transparent
        ax.imshow(overlay)

        # context patches: no overlay (transparent, show original frame)
        label = f'Frame {f_orig}'
        if f_orig == cf: label += ' ★COLLISION'
        ax.set_title(label, fontsize=10)
        ax.axis('off')

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=(0.1, 0.4, 0.9, 0.6), label='Target (masked)')]
    fig.legend(handles=legend, loc='lower center', fontsize=12)
    fig.suptitle(f'{variant} — collision at frame {cf}', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tensor_dir', default='/research/d7/spc/yrwang5/JEPA_experiements/data/clevrer/tensors')
    parser.add_argument('--ann_dir', default='/research/d7/spc/yrwang5/JEPA_experiements/data/clevrer/processed_proposals')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--num', type=int, default=3)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = 'imgs'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cls = MODEL_MAP[args.variant]
    model = model_cls(img_size=96, num_frames=16, mask_ratio=0.6).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)

    videos = find_collision_videos(args.ann_dir, args.num)
    for i, (vid_idx, cf) in enumerate(videos):
        vpath = os.path.join(args.tensor_dir, f'video_{vid_idx:05d}.pth')
        out_name = f'{args.variant}_mask_visualization_{i+1:02d}.png'
        out_path = os.path.join(args.out_dir, out_name)
        visualize_one(args.variant, vpath, cf, model, device, out_path)
        print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
