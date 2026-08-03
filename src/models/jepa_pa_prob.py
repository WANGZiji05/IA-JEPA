"""
PA-Probabilistic JEPA.

Instead of deterministic top-K by physics importance (original PA-Masking),
samples target patches from an importance-weighted distribution.
High-physics patches are more likely to be masked, but never deterministically —
this preserves the scattered nature that prevents whole-object occlusion.
"""

import torch
from src.models.jepa_physics_aware import PhysicsAwareJEPA


class PAProbabilisticJEPA(PhysicsAwareJEPA):
    """PA masking with probabilistic sampling instead of hard top-K."""

    def get_physics_mask(self, videos):
        B, C, T, H, W = videos.shape
        device = videos.device

        Td = T // self.tubelet_size
        Hp, Wp = H // self.patch_size, W // self.patch_size
        N = Td * Hp * Wp

        K_target = int(N * (1.0 - self.mask_ratio))
        K_context = N - K_target

        # PA importance pipeline (inherited)
        vel, acc = self._extract_motion(videos)
        importance = self._compute_importance(vel, acc)
        importance_flat = importance.reshape(B, N)

        # Convert to sampling weights (not hard threshold)
        temperature = 0.3
        weights = torch.softmax(importance_flat / temperature, dim=1)

        # Sample K_target without replacement, weighted by importance
        target_idx_list = []
        for b in range(B):
            sampled = torch.multinomial(weights[b], K_target, replacement=False)
            target_idx_list.append(sampled)
        target_idx = torch.stack(target_idx_list)

        # context = remaining
        ctx_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        ctx_mask.scatter_(1, target_idx, False)
        all_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        context_idx = all_idx[ctx_mask].reshape(B, K_context)

        return context_idx, target_idx
