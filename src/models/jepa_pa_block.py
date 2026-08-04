"""
PA-Block JEPA.

Multi-block masking with block center placement biased by PA importance.
Collision-heavy regions attract more blocks, but blocks remain random-sized
and well-spaced — preventing the whole-object occlusion that kills
Object/Interaction variants.
"""

import torch
from src.models.jepa_physics_aware import PhysicsAwareJEPA


class PABlockJEPA(PhysicsAwareJEPA):
    """Multi-block + PA-weighted center placement."""

    def generate_pa_blocks(self, videos, Td, Hp, Wp, importance_3d):
        """Place 3D blocks with PA-weighted centers."""
        B, _, _, _, _ = videos.shape
        device = videos.device
        N = Td * Hp * Wp
        K_target = int(N * (1.0 - self.mask_ratio))

        # Merge PA importance across time → spatial placement prior
        # importance_3d: (B, Td, Hp, Wp)
        spatial_importance = importance_3d.mean(dim=1).reshape(B, -1)  # (B, 36)
        spatial_prob = torch.softmax(spatial_importance / 0.5, dim=1)  # (B, 36)

        max_t, max_h, max_w = min(3, Td), min(3, Hp), min(3, Wp)
        S = Hp * Wp  # 36

        all_targets = []
        for b in range(B):
            mask_3d = torch.zeros(Td, Hp, Wp, dtype=torch.bool, device=device)
            for _ in range(100):  # overshoot, no scattered fill
                if mask_3d.sum() >= K_target + 10:  # overshoot margin
                    break
                bt = torch.randint(1, max_t + 1, (1,), device=device).item()
                bh = torch.randint(1, max_h + 1, (1,), device=device).item()
                bw = torch.randint(1, max_w + 1, (1,), device=device).item()
                t0 = torch.randint(0, Td - bt + 1, (1,), device=device).item()

                spatial_center = torch.multinomial(spatial_prob[b], 1).item()
                cy, cx = divmod(spatial_center, Wp)
                h0 = max(0, min(cy - bh // 2, Hp - bh))
                w0 = max(0, min(cx - bw // 2, Wp - bw))

                mask_3d[t0:t0+bt, h0:h0+bh, w0:w0+bw] = True

            flat = mask_3d.reshape(-1)
            target = torch.where(flat)[0]         # all block-covered positions
            target = target[:K_target]            # trim, preserves contiguity
            all_targets.append(target)

        target_idx = torch.stack(all_targets)
        all_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        ctx_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        ctx_mask.scatter_(1, target_idx, False)
        context_idx = all_idx[ctx_mask].reshape(B, N - K_target)
        return context_idx, target_idx

    def forward(self, videos):
        B, C, T, H, W = videos.shape
        device = videos.device
        Td = T // self.tubelet_size
        Hp, Wp = H // self.patch_size, W // self.patch_size

        vel, acc = self._extract_motion(videos)
        importance = self._compute_importance(vel, acc)  # (B, Td, Hp, Wp)

        context_idx, target_idx = self.generate_pa_blocks(
            videos, Td, Hp, Wp, importance
        )

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
