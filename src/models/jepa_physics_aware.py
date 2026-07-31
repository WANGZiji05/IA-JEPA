"""
Physics-Aware Masking JEPA (PA-JEPA) — Pure Self-Supervised.

Three improvements over IA-JEPA's Interaction-Aware masking, each addressing
a specific limitation:

1. Multi-scale motion (velocity + acceleration)
   IA-JEPA uses only acceleration (diff²) → misses constant-velocity motion.
   PA adds velocity (diff¹) with equal weight, capturing both sustained motion
   and sudden changes.

2. True spatiotemporal importance (per-tubelet, full rank)
   IA-JEPA factorises importance as spatial × temporal (rank-1 separable):
       imp[t,h,w] = spatial[h,w] × temporal[t]
   This cannot represent "this specific patch at this specific moment."
   PA scores each tubelet independently → imp[t,h,w] is a full-rank tensor.

3. Local contrast normalisation
   IA-JEPA has no normalisation — global camera motion (panning, shake)
   produces uniformly high diff values everywhere, drowning the signal.
   PA subtracts the local spatial mean and takes ReLU:
       contrast[h,w] = relu(imp[h,w] - avg_pool_3×3(imp)[h,w])
   This suppresses global motion while preserving locally anomalous patches
   (the actual physical interactions).

All three signals come exclusively from pixel data — zero external annotations.
"""

import torch
import torch.nn.functional as F

from src.models.jepa_baseline import VideoJEPA, update_target_encoder  # noqa: F401


class PhysicsAwareJEPA(VideoJEPA):
    """PA-JEPA: pure-SSL physics-aware masked video prediction.

    Inherits VideoJEPA backbone (patch embed, context/target encoders,
    EMA, predictor).  Only the mask-generation logic is replaced.

    Forward signature:  model(video)  — same as baseline, no external inputs.
    """

    def __init__(
        self,
        img_size=112,
        patch_size=16,
        num_frames=16,
        tube_size=2,
        embed_dim=192,
        enc_depth=6,
        pred_depth=3,
        num_heads=6,
        mask_ratio=0.6,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tube_size=tube_size,
            embed_dim=embed_dim,
            enc_depth=enc_depth,
            pred_depth=pred_depth,
            num_heads=num_heads,
            mask_ratio=mask_ratio,
        )

    # ------------------------------------------------------------------
    # 1. Multi-scale motion extraction
    # ------------------------------------------------------------------
    def _extract_motion(self, videos):
        """Compute per-tubelet velocity and acceleration maps.

        Args:
            videos: [B, C, T, H, W]  (normalised frames)

        Returns:
            vel: [B, Td, Hp, Wp]  first-order temporal difference
            acc: [B, Td, Hp, Wp]  second-order temporal difference
        """
        B, C, T, H, W = videos.shape

        # pixel-level temporal differences
        diff1 = torch.abs(videos[:, :, 1:] - videos[:, :, :-1])       # (B,C,T-1,H,W)
        diff2 = torch.abs(diff1[:, :, 1:] - diff1[:, :, :-1])         # (B,C,T-2,H,W)

        # average over colour channels
        vel = diff1.mean(dim=1)  # (B, T-1, H, W)
        acc = diff2.mean(dim=1)  # (B, T-2, H, W)

        # pad to full T frames (replicate edge values)
        vel = F.pad(vel, (0, 0, 0, 0, 0, 1), mode='replicate')
        acc = F.pad(acc, (0, 0, 0, 0, 1, 1), mode='replicate')

        # compute grid dimensions dynamically (not hard-coded)
        Hp, Wp = H // self.patch_size, W // self.patch_size
        Td = T // self.tubelet_size

        # spatial pool → patch grid
        vel_p = vel.reshape(B, T, Hp, self.patch_size, Wp, self.patch_size).mean(dim=(3, 5))
        acc_p = acc.reshape(B, T, Hp, self.patch_size, Wp, self.patch_size).mean(dim=(3, 5))

        # temporal pool → tubelet grid
        vel_t = vel_p.reshape(B, Td, self.tubelet_size, Hp, Wp).mean(dim=2)
        acc_t = acc_p.reshape(B, Td, self.tubelet_size, Hp, Wp).mean(dim=2)

        return vel_t, acc_t   # each (B, Td, Hp, Wp)

    # ------------------------------------------------------------------
    # 2. Importance computation with local contrast normalisation
    # ------------------------------------------------------------------
    def _compute_importance(self, vel, acc):
        """Fuse velocity + acceleration, then apply local contrast normalisation.

        Pipeline:
          1. Per-video min-max normalise each modality independently.
          2. Equal-weight fusion: imp = 0.5 * vel_norm + 0.5 * acc_norm.
          3. Local contrast: subtract 3×3 spatial mean, take ReLU, re-normalise.
        """
        # --- per-video min-max (handles brightness / scene variation) ---
        def _norm(x):
            B = x.shape[0]
            xf = x.reshape(B, -1)
            vmin = xf.min(dim=1, keepdim=True).values.reshape(B, 1, 1, 1)
            vmax = xf.max(dim=1, keepdim=True).values.reshape(B, 1, 1, 1)
            return (x - vmin) / (vmax - vmin).clamp(min=1e-8)   # all-zero → 0 safely

        imp = 0.5 * _norm(vel) + 0.5 * _norm(acc)   # (B, Td, Hp, Wp)

        # --- local contrast: suppress uniform global motion ---
        B, Td, Hp, Wp = imp.shape
        flat = imp.reshape(B * Td, 1, Hp, Wp)
        local_mean = F.avg_pool2d(flat, kernel_size=3, stride=1, padding=1)
        local_mean = local_mean.reshape(B, Td, Hp, Wp)

        contrast = torch.relu(imp - local_mean)        # keep only locally anomalous

        # re-normalise per video
        cf = contrast.reshape(B, -1)
        denom = cf.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        contrast = contrast / denom.reshape(B, 1, 1, 1)

        return contrast   # (B, Td, Hp, Wp)

    # ------------------------------------------------------------------
    # 3. Physics-aware mask generation
    # ------------------------------------------------------------------
    def get_physics_mask(self, videos):
        """Pure physics-aware mask: low-importance → context, high-importance → target.

        The encoder sees background / low-motion regions and must predict the
        representations of interaction-heavy patches.  This is the original IA-JEPA
        masking philosophy — force the model to infer physics from context.
        """
        B, C, T, H, W = videos.shape
        device = videos.device

        Td = T // self.tubelet_size
        Hp, Wp = H // self.patch_size, W // self.patch_size
        N = Td * Hp * Wp

        K_target = int(N * (1.0 - self.mask_ratio))
        K_context = N - K_target

        # --- physics importance pipeline ---
        vel, acc = self._extract_motion(videos)
        importance = self._compute_importance(vel, acc)          # (B, Td, Hp, Wp)
        importance_flat = importance.reshape(B, N)               # (B, N)
        _, sorted_idx = torch.sort(importance_flat, dim=1, descending=True)

        target_idx = sorted_idx[:, :K_target]                    # high → predict
        context_idx = sorted_idx[:, K_target:]                   # low → visible

        return context_idx, target_idx

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, videos):
        """Forward following the Object / Interaction variant pattern.

        Args:
            videos: [B, C, T, H, W]

        Returns:
            pred_latents:   [B, K_tgt, E]  predicted representations
            target_latents: [B, K_tgt, E]  target (EMA) representations
        """
        B = videos.shape[0]
        device = videos.device

        context_idx, target_idx = self.get_physics_mask(videos)

        patches = self.patch_embed(videos)                       # (B, N, E)
        batch_idx = torch.arange(B, device=device).unsqueeze(1)

        # --- context: visible, low-importance patches ---
        ctx_patches = patches[batch_idx, context_idx]            # (B, K_ctx, E)
        ctx_latents = self.context_encoder(ctx_patches)          # (B, K_ctx, E)

        # --- target: high-importance patches to predict ---
        with torch.no_grad():
            tgt_full = self.target_encoder(patches)              # (B, N, E)
            tgt_latents = tgt_full[batch_idx, target_idx]       # (B, K_tgt, E)

        # --- predict ---
        pred = self.predictor(ctx_latents)                       # (B, K_ctx, E)
        if pred.shape[1] != tgt_latents.shape[1]:
            pred = pred[:, :tgt_latents.shape[1], :]            # (B, K_tgt, E)

        return pred, tgt_latents
