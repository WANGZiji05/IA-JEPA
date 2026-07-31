"""
Mixed Physics-Aware JEPA (Mixed-PA).

Combines PA-Masking importance with random masking so the encoder sees
both physics-relevant regions AND diverse context.

Context = 50% high-importance (encoder SEES physics) + 50% random (diversity).
Target  = remaining patches.

Strategy rationale:
  Pure PA-Masking shows the encoder only low-importance (background) patches,
  starving it of direct exposure to collisions and motion.  Mixed-PA puts
  physics patches *in context* so the encoder learns to represent them, while
  random sampling preserves the full-scene diversity that benefits downstream
  causal reasoning.
"""

import torch
from src.models.jepa_physics_aware import PhysicsAwareJEPA


class MixedPhysicsAwareJEPA(PhysicsAwareJEPA):
    """PA-JEPA with mixed context: physics-guided + random.

    Inherits the full PA motion pipeline (vel+acc, local contrast, etc.)
    and only changes how context indices are assembled.
    """

    def get_physics_mask(self, videos):
        """Mixed context: 50% physics (high-importance visible) + 50% random."""
        B, C, T, H, W = videos.shape
        device = videos.device

        Td = T // self.tubelet_size
        Hp, Wp = H // self.patch_size, W // self.patch_size
        N = Td * Hp * Wp

        K_context = int(N * self.mask_ratio)
        K_target = N - K_context
        K_half_ctx = K_context // 2

        # --- physics importance pipeline (inherited from PhysicsAwareJEPA) ---
        vel, acc = self._extract_motion(videos)
        importance = self._compute_importance(vel, acc)
        importance_flat = importance.reshape(B, N)
        _, sorted_idx = torch.sort(importance_flat, dim=1, descending=True)

        # --- context: half physics, half random ---
        ctx_physics = sorted_idx[:, :K_half_ctx]                  # highest physics

        low_pool = sorted_idx[:, K_half_ctx:]                     # remaining patches
        rand_perm = torch.argsort(
            torch.rand(B, low_pool.shape[1], device=device), dim=1,
        )
        ctx_random = low_pool.gather(1, rand_perm[:, :K_half_ctx])

        context_idx = torch.cat([ctx_physics, ctx_random], dim=1)

        # --- target = everything else ---
        ctx_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        ctx_mask.scatter_(1, context_idx, False)
        target_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)[ctx_mask]
        target_idx = target_idx.reshape(B, K_target)

        return context_idx, target_idx
