"""
Random-Tube Baseline (V-JEPA family).

Masks entire temporal tubes — same spatial position across all frames shares
the same fate.  36 spatial positions × 8 tubelets each → 288 tubelets total.
40% of spatial positions → target (all 8 tubelets of those positions).
"""

import torch
from src.models.jepa_baseline import VideoJEPA


class RandomTubeJEPA(VideoJEPA):
    """Random tube masking: spatial position = minimum masking unit."""

    def generate_mask(self, B, N, device):
        """Mask by spatial position (tube), not individual tubelet."""
        # Compute grid: N = Td × Sd, Td = num_frames // tubelet_size
        Td = self.pos_embed.shape[1] // 36  # 288 / 36 = 8  (since Sd=36 for 96/16)
        Sd = self.pos_embed.shape[1] // Td   # 288 / 8 = 36
        N_spatial = Sd

        # Shuffle spatial positions, not tubelets
        noise = torch.rand(B, N_spatial, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)

        # target = (1 - mask_ratio) of spatial positions → expand to tubelets
        K_tubes_target = int(N_spatial * (1.0 - self.mask_ratio))  # ~14
        target_spatial = ids_shuffle[:, :K_tubes_target]            # (B, 14)
        context_spatial = ids_shuffle[:, K_tubes_target:]            # (B, 22)

        # Expand to tubelet indices: each spatial position → Td tubelets
        target_idx = torch.cat(
            [target_spatial + t * N_spatial for t in range(Td)], dim=1
        )
        context_idx = torch.cat(
            [context_spatial + t * N_spatial for t in range(Td)], dim=1
        )

        # Return in IA-JEPA convention: context, target
        return context_idx, target_idx

    def forward(self, videos):
        B = videos.shape[0]
        device = videos.device
        context_idx, target_idx = self.generate_mask(B, 0, device)

        patches = self.patch_embed(videos)
        batch_idx = torch.arange(B, device=device).unsqueeze(1)

        ctx = patches[batch_idx, context_idx]
        ctx_latents = self.context_encoder(ctx)

        with torch.no_grad():
            tgt_full = self.target_encoder(patches)
            tgt_latents = tgt_full[batch_idx, target_idx]

        pred = self.predictor(ctx_latents)
        if pred.shape[1] != tgt_latents.shape[1]:
            pred = pred[:, :tgt_latents.shape[1], :]
        return pred, tgt_latents
