"""
Batch-convert CLEVRER mp4 videos to .pth tensors.

Source:  videos/{split}/video_XXXXX.mp4   (128 frames, 320×480)
Output:  {out_dir}/video_XXXXX.pth        (uint8 tensor [3, 128, H, W])

Usage:
    python scripts/preprocess_videos.py \
        --src /path/to/clevrer/videos \
        --split train \
        --frame-size 96 \
        --out data/clevrer_tensors

On a server with a decent CPU, ~10000 videos take 30–60 minutes.
GPU is not needed — torchvision CPU decoding works fine.
"""

import argparse
import os
import torch
import torchvision
from tqdm import tqdm


def process_one(mp4_path, out_path, frame_size):
    """Decode one mp4 → save as [C, T, H, W] uint8 tensor."""
    frames, _, info = torchvision.io.read_video(
        mp4_path, output_format='TCHW', pts_unit='sec'
    )
    # frames: (T, C, H, W) uint8

    if frame_size is not None and frames.shape[-1] != frame_size:
        frames = frames.float()
        frames = torch.nn.functional.interpolate(
            frames.permute(1, 0, 2, 3),          # (C, T, H, W)
            size=frame_size,
            mode='bilinear',
            antialias=True,
        ).permute(1, 0, 2, 3)                    # back to (T, C, H, W)
        frames = frames.clamp(0, 255).to(torch.uint8)

    # save as [C, T, H, W] uint8 (IA-JEPA format)
    tensor = frames.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)
    torch.save(tensor, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='path to clevrer/videos')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--frame-size', type=int, default=96)
    parser.add_argument('--out', required=True, help='output directory')
    parser.add_argument('--start', type=int, default=0, help='first video index')
    parser.add_argument('--end', type=int, default=None, help='last video index (exclusive)')
    args = parser.parse_args()

    src_dir = os.path.join(args.src, args.split)
    out_dir = os.path.join(args.out, '')
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(src_dir) if f.endswith('.mp4'))
    if args.end is not None:
        files = files[args.start:args.end]
    else:
        files = files[args.start:]

    print(f"Processing {len(files)} videos: {args.src}/{args.split} → {out_dir}")
    print(f"Resize to: {args.frame_size}×{args.frame_size}")

    for fname in tqdm(files):
        idx = int(fname.replace('video_', '').replace('.mp4', ''))
        mp4_path = os.path.join(src_dir, fname)
        out_path = os.path.join(out_dir, f"video_{idx:05d}.pth")

        if os.path.exists(out_path):
            continue  # skip already-processed

        try:
            process_one(mp4_path, out_path, args.frame_size)
        except Exception as e:
            tqdm.write(f"Error on {fname}: {e}")


if __name__ == '__main__':
    main()
