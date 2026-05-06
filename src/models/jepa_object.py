import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.jepa_baseline import VideoJEPA, update_target_encoder

class ObjectMaskedJEPA(VideoJEPA):
    """
    Video JEPA variant that masks patches based on object locations (from masks) or motion.
    """
    def __init__(self, img_size=112, patch_size=16, num_frames=16, mask_ratio=0.6):
        super().__init__(img_size=img_size, patch_size=patch_size, 
                         num_frames=num_frames, mask_ratio=mask_ratio)
        
    def get_mask_indices(self, videos, masks=None):
        """
        videos: [B, C, T, H, W]
        masks: [B, 1, T, H, W]
        returns: context_idx, target_idx (spatial indices)
        """
        B, C, T, H, W = videos.shape
        
        # Aggregate masks across time
        mask_spatial = masks.max(dim=2)[0] # [B, 1, H, W]
        mask_importance = F.avg_pool2d(mask_spatial, self.patch_size).squeeze(1) # [B, grid, grid]

        # Motion Fallback for samples with no masks
        diff = torch.abs(videos[:, :, 1:] - videos[:, :, :-1])
        motion = diff.mean(dim=(1, 2))
        motion_importance = F.avg_pool2d(motion.unsqueeze(1), self.patch_size).squeeze(1)

        # Decide per-sample: if mask is all zero, use motion
        mask_exists = (mask_importance.sum(dim=(1, 2)) > 0).view(B, 1, 1)
        importance_grid = torch.where(mask_exists, mask_importance, motion_importance)

        # Flatten and Mask
        importance_flat = importance_grid.view(B, -1)
        num_patches = importance_flat.shape[1]
        num_keep = int(num_patches * (1 - self.mask_ratio))
        
        # Target = High Importance (Objects), Context = Low Importance (Background)
        _, indices = torch.sort(importance_flat, dim=1, descending=True)
        context_idx = indices[:, num_keep:]
        target_idx = indices[:, :num_keep]
        
        return context_idx, target_idx

    def forward(self, videos, masks=None):
        B, C, T, H, W = videos.shape
        device = videos.device
        
        context_idx, target_idx = self.get_mask_indices(videos, masks)
        
        patches = self.patch_embed(videos)
        T_down = T // self.patch_embed.tubelet_size
        S_down = (H // self.patch_size) * (W // self.patch_size)
        
        # Spatiotemporal expansion
        full_context_idx = torch.cat([context_idx + t * S_down for t in range(T_down)], dim=1)
        full_target_idx = torch.cat([target_idx + t * S_down for t in range(T_down)], dim=1)

        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        context_patches = patches[batch_idx, full_context_idx]
        context_latents = self.context_encoder(context_patches)
        
        with torch.no_grad():
            target_latents = self.target_encoder(patches)
            target_latents = target_latents[batch_idx, full_target_idx]
            
        pred_latents = self.predictor(context_latents) 
        if pred_latents.shape[1] != target_latents.shape[1]:
            pred_latents = pred_latents[:, :target_latents.shape[1], :]
            
        return pred_latents, target_latents
