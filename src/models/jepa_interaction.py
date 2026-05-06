import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.jepa_object import ObjectMaskedJEPA

class InteractionAwareJEPA(ObjectMaskedJEPA):
    """
    Video JEPA variant that masks patches based on 'Action Intensity' (acceleration)
    and ground-truth collision events.
    """
    def __init__(self, img_size=112, patch_size=16, num_frames=16, mask_ratio=0.6):
        super().__init__(img_size=img_size, patch_size=patch_size, 
                         num_frames=num_frames, mask_ratio=mask_ratio)
        
    def get_interaction_mask(self, videos, collision_frames=None):
        """
        videos: [B, C, T, H, W]
        collision_frames: List of lists or tensor [B, max_collisions]
        returns: context_idx, target_idx (spatiotemporal indices)
        """
        B, C, T, H, W = videos.shape
        device = videos.device
        
        # 1. Action Intensity (Acceleration) - Spatial component
        diff1 = torch.abs(videos[:, :, 1:] - videos[:, :, :-1])
        diff2 = torch.abs(diff1[:, :, 1:] - diff1[:, :, :-1])
        action_intensity = diff2.mean(dim=(1, 2)) # [B, H, W]
        action_grid = F.avg_pool2d(action_intensity.unsqueeze(1), self.patch_size).squeeze(1)
        
        # 2. Temporal component: prefer masking frames near collisions
        # This is a simplified version: we mask the WHOLE frame if it's near a collision
        T_down = T // self.patch_embed.tubelet_size
        S_down = (H // self.patch_size) * (W // self.patch_size)
        
        # Spatial importance from action_grid
        spatial_importance = action_grid.view(B, -1) # [B, S_down]
        
        # Identify "Interaction" patches
        # For now, let's use the collision_frames to boost temporal importance
        temporal_importance = torch.ones((B, T_down), device=device)
        if collision_frames is not None:
            # collision_frames contains indices in [0, 128)
            # Map them to [0, T_down)
            for b in range(B):
                for cf in collision_frames[b]:
                    if cf < 0: continue
                    t_idx = cf // self.patch_embed.tubelet_size
                    if t_idx < T_down:
                        temporal_importance[b, t_idx] *= 5.0 # Boost importance

        # Combine spatial and temporal importance
        # Spatiotemporal importance [B, T_down, S_down]
        st_importance = spatial_importance.unsqueeze(1) * temporal_importance.unsqueeze(2)
        st_importance = st_importance.view(B, -1) # [B, T_down * S_down]
        
        num_patches = st_importance.shape[1]
        num_keep = int(num_patches * (1 - self.mask_ratio))
        
        _, indices = torch.sort(st_importance, dim=1, descending=True)
        target_idx = indices[:, :num_keep]
        context_idx = indices[:, num_keep:]
        
        return context_idx, target_idx

    def forward(self, videos, collision_frames=None):
        B, C, T, H, W = videos.shape
        device = videos.device
        
        context_idx, target_idx = self.get_interaction_mask(videos, collision_frames)
        
        patches = self.patch_embed(videos)
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        
        context_patches = patches[batch_idx, context_idx]
        context_latents = self.context_encoder(context_patches)
        
        with torch.no_grad():
            target_latents = self.target_encoder(patches)
            target_latents = target_latents[batch_idx, target_idx]
            
        pred_latents = self.predictor(context_latents) 
        if pred_latents.shape[1] != target_latents.shape[1]:
            pred_latents = pred_latents[:, :target_latents.shape[1], :]
            
        return pred_latents, target_latents
