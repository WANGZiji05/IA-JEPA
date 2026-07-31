"""
Collision Expert Probe — binary classifier on frozen JEPA features.

"Does a collision occur in this clip?"
Directly measures whether the encoder has learned physical intuition.
"""

import torch
import torch.nn as nn


class CollisionExpertProbe(nn.Module):
    """Lightweight binary classifier on pooled encoder features."""

    def __init__(self, feature_dim=192, hidden_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        """features: (B, N, D) → mean pool → binary logit."""
        x = features.mean(dim=1)          # (B, D)
        return self.classifier(x).squeeze(-1)  # (B,)


def train_collision_probe(model, train_loader, val_loader, device, epochs=10):
    """Train CollisionExpertProbe on frozen encoder features."""
    probe = CollisionExpertProbe(model.embed_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        probe.train()
        for video, label in train_loader:
            video, label = video.to(device), label.to(device)
            with torch.no_grad():
                feats = model.context_encoder(
                    model.patch_embed(video)
                    + (model.pos_embed if hasattr(model, 'pos_embed') else 0)
                )
            opt.zero_grad()
            loss = loss_fn(probe(feats), label)
            loss.backward()
            opt.step()

        acc = evaluate_collision_probe(probe, model, val_loader, device)
        print(f"  Collision Probe Epoch {epoch + 1}: val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    if best_state is not None:
        probe.load_state_dict(best_state)
    return probe


@torch.no_grad()
def evaluate_collision_probe(probe, model, loader, device):
    probe.eval()
    correct, total = 0, 0
    for video, label in loader:
        video, label = video.to(device), label.to(device)
        feats = model.context_encoder(
            model.patch_embed(video)
            + (model.pos_embed if hasattr(model, 'pos_embed') else 0)
        )
        preds = (torch.sigmoid(probe(feats)) > 0.5).float()
        correct += (preds == label).sum().item()
        total += label.numel()
    return correct / total if total > 0 else 0
