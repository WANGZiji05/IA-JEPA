import torch
import torch.nn as nn
import math
import copy

class VideoPatchEmbed(nn.Module):
    """ Converts a video tensor into a sequence of patch embeddings. """
    def __init__(self, img_size=112, patch_size=16, tube_size=2, in_chans=3, embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tube_size
        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(tube_size, patch_size, patch_size), 
            stride=(tube_size, patch_size, patch_size)
        )

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        x = self.proj(x)  # (B, E, T//tube_size, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x

class LightweightViT(nn.Module):
    """ A simple Transformer encoder used for Context/Target encoders and Predictor. """
    def __init__(self, embed_dim=192, depth=4, num_heads=6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            batch_first=True, 
            activation='gelu',
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.norm(x)

class VideoJEPA(nn.Module):
    """ Patch-Masked Video JEPA Baseline """
    def __init__(self, img_size=112, patch_size=16, num_frames=16, tube_size=2,
                 embed_dim=192, enc_depth=6, pred_depth=3, num_heads=6, mask_ratio=0.6):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.tubelet_size = tube_size
        self.embed_dim = embed_dim
        
        # 1. Patch Embedding
        self.patch_embed = VideoPatchEmbed(img_size, patch_size, tube_size, 3, embed_dim)
        
        # Calculate sequence length N = (T / tube_size) * (H / patch_size) * (W / patch_size)
        num_patches = (num_frames // tube_size) * (img_size // patch_size) ** 2
        
        # Learnable Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 2. Context Encoder (processes visible patches)
        self.context_encoder = LightweightViT(embed_dim, enc_depth, num_heads)
        
        # 3. Target Encoder (processes all patches, updated via EMA)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False  # Target encoder is stop-gradient
            
        # 4. Predictor (processes context latents + mask tokens)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        self.predictor = LightweightViT(embed_dim, pred_depth, num_heads)

    def generate_mask(self, B, N, device):
        """ Generates random masks. Returns indices to keep and indices to mask. """
        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        len_keep = int(N * (1 - self.mask_ratio))
        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]
        
        return ids_keep, ids_mask

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        Returns: predicted mask latents, and target mask latents
        """
        B = x.shape[0]
        
        # Embed patches and add absolute positional embeddings
        x_patches = self.patch_embed(x) # (B, N, E)
        x_patches = x_patches + self.pos_embed
        N = x_patches.shape[1]
        
        # Generate mask indices
        ids_keep, ids_mask = self.generate_mask(B, N, x.device)
        
        # ==================== TARGET PIPELINE ====================
        with torch.no_grad():
            # Target encoder processes full sequence (standard JEPA design)
            target_latents_full = self.target_encoder(x_patches)
            # We only care about predicting the masked regions
            # Gather targets corresponding to the masked positions
            batch_indices = torch.arange(B).unsqueeze(-1).expand(-1, ids_mask.size(1)).to(x.device)
            target_latents = target_latents_full[batch_indices, ids_mask] # (B, len_mask, E)
            
        # ==================== CONTEXT PIPELINE ====================
        # Gather visible patches for context
        batch_indices_keep = torch.arange(B).unsqueeze(-1).expand(-1, ids_keep.size(1)).to(x.device)
        x_context = x_patches[batch_indices_keep, ids_keep] # (B, len_keep, E)
        
        # Encode context
        context_latents = self.context_encoder(x_context)
        
        # ==================== PREDICTOR PIPELINE ====================
        # Create mask tokens for the predictor
        mask_tokens = self.mask_token.repeat(B, ids_mask.size(1), 1)
        
        # Add corresponding positional embeddings to context and mask tokens
        pos_keep = self.pos_embed.expand(B, -1, -1)[batch_indices_keep, ids_keep]
        pos_mask = self.pos_embed.expand(B, -1, -1)[batch_indices, ids_mask]
        
        # Concatenate context latents and mask tokens
        pred_input = torch.cat([context_latents + pos_keep, mask_tokens + pos_mask], dim=1)
        
        # Run predictor
        pred_output = self.predictor(pred_input)
        
        # The predictor outputs representations for [context + mask]. 
        # We slice out the predictions corresponding to the mask tokens (the latter part).
        pred_latents = pred_output[:, ids_keep.size(1):, :] # (B, len_mask, E)
        
        return pred_latents, target_latents

def update_target_encoder(context_encoder, target_encoder, momentum=0.996):
    """ Exponential Moving Average update for the target encoder. """
    with torch.no_grad():
        for param_q, param_k in zip(context_encoder.parameters(), target_encoder.parameters()):
            param_k.data.mul_(momentum).add_((1.0 - momentum) * param_q.detach().data)
