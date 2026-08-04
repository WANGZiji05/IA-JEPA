"""
Multi-Block JEPA (V-JEPA style, adapted for IA-JEPA's 288-tubelet grid).

Places npred random 3D spatiotemporal blocks whose size is sampled once
per batch from continuous scale/aspect_ratio ranges.  Blocks can overlap;
the union of all block-covered tubelets becomes the prediction target.

Match IA-JEPA convention: target = (1 - mask_ratio) of tubelets.
"""

import math
import torch
from src.models.jepa_baseline import VideoJEPA


class MultiBlockJEPA(VideoJEPA):
    """Random 3D block masking following V-JEPA's multi-block logic."""

    # ---- block size sampling (V-JEPA style) ----
    def _sample_block_size(self, generator):
        """Sample one block's (t, h, w) in tubelet units.
        V-JEPA-style: sample temporal scale, spatial scale, aspect ratio
        from continuous ranges, then compute dimensions from area × ratio.
        """
        Td = self._Td
        Hp, Wp = self._Hp, self._Wp

        # temporal: 40-80% of frames (smaller than V-JEPA's 100% to leave temporal diversity)
        _rand = torch.rand(1, generator=generator).item()
        t = max(1, int(Td * (0.4 + _rand * 0.4)))
        t = min(t, Td)

        # spatial: 15-40% of the spatial grid
        _rand = torch.rand(1, generator=generator).item()
        spatial_scale = 0.15 + _rand * 0.25
        spatial_area = int(Hp * Wp * spatial_scale)

        # aspect ratio: 0.5 - 2.0
        _rand = torch.rand(1, generator=generator).item()
        ar = 0.5 + _rand * 1.5
        h = min(Hp, max(1, int(round(math.sqrt(spatial_area * ar)))))
        w = min(Wp, max(1, int(round(math.sqrt(spatial_area / ar)))))

        return (t, h, w)

    def generate_blocks(self, B, device):
        """Place random 3D blocks (always overshoot K_target, then trim)."""
        if not hasattr(self, '_step_counter'):
            self._step_counter = 0
        Td, Hp, Wp = self._Td, self._Hp, self._Wp
        N = Td * Hp * Wp
        K_target = int(N * (1.0 - self.mask_ratio))

        seed = self._step_counter
        self._step_counter += 1
        g = torch.Generator(device='cpu')
        g.manual_seed(seed)
        block_size = self._sample_block_size(g)

        # Place MORE blocks than needed — always overshoot, then trim.  No scattered fills.
        block_volume = block_size[0] * block_size[1] * block_size[2]
        npred = max(1, K_target // max(1, block_volume)) + 3  # +3 guarantees overshoot

        all_targets = []
        for _ in range(B):
            mask_3d = torch.ones(Td, Hp, Wp, dtype=torch.int32, device=device)
            for _ in range(npred):
                t, h, w = block_size
                t0 = torch.randint(0, Td - t + 1, (1,), device=device).item()
                h0 = torch.randint(0, Hp - h + 1, (1,), device=device).item()
                w0 = torch.randint(0, Wp - w + 1, (1,), device=device).item()
                mask_3d[t0:t0+t, h0:h0+h, w0:w0+w] = 0

            flat = mask_3d.reshape(-1)
            target = torch.where(flat == 0)[0]  # all block-covered positions
            target = target[:K_target]           # trim to exactly K_target (keeps contiguity)
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
        self._Td = T // self.tubelet_size
        self._Hp, self._Wp = H // self.patch_size, W // self.patch_size

        if not hasattr(self, '_step_counter'):
            self._step_counter = 0

        context_idx, target_idx = self.generate_blocks(B, device)

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
