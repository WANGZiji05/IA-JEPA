import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import os
import gc

class MultimodalChoiceProbe(nn.Module):
    """
    Definitive Multimodal Probe for CLEVRER.
    Fuses JEPA video patches with Question and Choice text.
    """
    def __init__(self, feature_dim, vocab_size=2000, text_dim=256, hidden_dim=512, num_descriptive_classes=41):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, text_dim)
        self.text_gru = nn.GRU(text_dim, text_dim, batch_first=True)
        self.pos_enc = nn.Parameter(torch.randn(1, 8, feature_dim))
        
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.descriptive_fusion = nn.Sequential(
            nn.Linear(hidden_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_descriptive_classes)
        )
        self.choice_reasoner = nn.Sequential(
            nn.Linear(hidden_dim + text_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.word_map = {"<pad>": 0, "<unk>": 1}

    def encode_text(self, text_list, device):
        batch_ids = []
        max_len = 0
        for text in text_list:
            t_str = str(text).lower().replace('?', '').strip()
            words = t_str.split()
            ids = []
            for w in words:
                if w not in self.word_map and len(self.word_map) < 1999:
                    self.word_map[w] = len(self.word_map)
                ids.append(self.word_map.get(w, 1))
            
            if not ids: ids = [0]
            batch_ids.append(ids)
            max_len = max(max_len, len(ids))
        
        padded = [ids + [0]*(max_len - len(ids)) for ids in batch_ids]
        x = torch.tensor(padded, dtype=torch.long, device=device)
        emb = self.embedding(x)
        _, h = self.text_gru(emb)
        return h.squeeze(0)

    def forward(self, features, questions, choices, device):
        B = features.shape[0]
        # features shape: [B, N, D]
        # JEPA typically has 8 temporal tokens. Spatial grid is N // 8.
        time_dim = 8 if features.shape[1] % 8 == 0 else (16 if features.shape[1] % 16 == 0 else 1)
        spatial_grid = features.shape[1] // time_dim
        
        # 1. Video Path: Reshape and Pool Spatially
        # [B, Time, Space, Dim]
        v = features.view(B, time_dim, spatial_grid, -1).mean(dim=2) # [B, Time, Dim]
        
        if v.shape[1] == 8:
            v = v + self.pos_enc
            
        v = v.permute(0, 2, 1) # [B, Dim, Time]
        v_feat = self.temporal_conv(v).squeeze(-1)
        
        # 2. Text Path
        q_feat = self.encode_text(questions, device)
        desc_logits = self.descriptive_fusion(torch.cat([v_feat, q_feat], dim=1))
        
        mc_logits = []
        for i in range(4):
            choice_batch = [c[i] for c in choices]
            c_feat = self.encode_text(choice_batch, device)
            score = self.choice_reasoner(torch.cat([v_feat, q_feat, c_feat], dim=1))
            mc_logits.append(score)
        return desc_logits, torch.cat(mc_logits, dim=1)

def train_probe_on_features(probe, train_loader, val_loader, device, epochs=5):
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    for epoch in range(epochs):
        probe.train()
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")
        for f, l, t, idx, q, choices in pbar:
            f = f.to(device)
            optimizer.zero_grad()
            desc_out, mc_out = probe(f, q, choices, device)
            desc_mask = torch.tensor([task == 'descriptive' for task in t], device=device)
            mc_mask = ~desc_mask
            loss = 0; count = 0
            if desc_mask.any():
                targets = torch.stack([l[i] for i, m in enumerate(desc_mask) if m]).to(device).long()
                loss += F.cross_entropy(desc_out[desc_mask], targets); count += 1
            if mc_mask.any():
                targets = torch.stack([l[i] for i, m in enumerate(mc_mask) if m]).to(device).float()
                loss += F.binary_cross_entropy_with_logits(mc_out[mc_mask], targets); count += 1
            if count > 0: (loss/count).backward(); optimizer.step()
        
        metrics = evaluate_probe_on_features(probe, val_loader, device)
        print(f"Epoch {epoch} Val Metrics: {metrics}", flush=True)
        gc.collect()
    return probe

@torch.no_grad()
def evaluate_probe_on_features(probe, val_loader, device):
    probe.eval()
    res = {'descriptive': {'correct': 0, 'total': 0}, 'mc_tasks': {'correct': 0, 'total': 0}}
    for f, l, t, idx, q, choices in val_loader:
        f = f.to(device)
        desc_out, mc_out = probe(f, q, choices, device)
        for i, task in enumerate(t):
            target = l[i].to(device)
            if task == 'descriptive':
                res['descriptive']['correct'] += (desc_out[i].argmax() == target).item()
                res['descriptive']['total'] += 1
            else:
                preds = (torch.sigmoid(mc_out[i]) > 0.5).float()
                res['mc_tasks']['correct'] += (preds == target).all().float().item()
                res['mc_tasks']['total'] += 1
    return {'descriptive_acc': res['descriptive']['correct']/(res['descriptive']['total']+1e-6),
            'mc_acc': res['mc_tasks']['correct']/(res['mc_tasks']['total']+1e-6)}

def train_probe(jepa_model, train_loader, val_loader, device, epochs=5):
    probe = MultimodalChoiceProbe(jepa_model.embed_dim).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    for epoch in range(epochs):
        probe.train()
        pbar = tqdm.tqdm(train_loader, desc=f"Video Epoch {epoch}")
        for v, l, t, idx, q, choices in pbar:
            v = v.to(device)
            with torch.no_grad():
                features = jepa_model.context_encoder(jepa_model.patch_embed(v) + (jepa_model.pos_embed if hasattr(jepa_model, 'pos_embed') else 0))
            optimizer.zero_grad()
            desc_out, mc_out = probe(features, q, choices, device)
            desc_mask = torch.tensor([task == 'descriptive' for task in t], device=device)
            mc_mask = ~desc_mask
            loss = 0; count = 0
            if desc_mask.any():
                targets = torch.stack([l[i] for i, m in enumerate(desc_mask) if m]).to(device).long()
                loss += F.cross_entropy(desc_out[desc_mask], targets); count += 1
            if mc_mask.any():
                targets = torch.stack([l[i] for i, m in enumerate(mc_mask) if m]).to(device).float()
                loss += F.binary_cross_entropy_with_logits(mc_out[mc_mask], targets); count += 1
            if count > 0: (loss/count).backward(); optimizer.step()
        metrics = evaluate_probe(jepa_model, probe, val_loader, device)
        print(f"Epoch {epoch} Val Metrics: {metrics}", flush=True)
        gc.collect()
    return probe

@torch.no_grad()
def evaluate_probe(jepa_model, probe, val_loader, device):
    return evaluate_probe_on_features(probe, val_loader, device)

class TemporalCausalProbe(MultimodalChoiceProbe): pass
