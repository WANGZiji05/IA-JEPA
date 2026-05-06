import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.clevrer_qa_dataset import CLEVRERQADataset, get_clevrer_qa_loaders
from src.eval.qa_probe import train_probe, evaluate_probe, MultimodalChoiceProbe, train_probe_on_features, evaluate_probe_on_features
from src.models.jepa_baseline import VideoJEPA
from src.models.jepa_object import ObjectMaskedJEPA
from src.models.jepa_interaction import InteractionAwareJEPA
import argparse
import os

def custom_collate(batch):
    clips = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    task_types = [item[2] for item in batch]
    video_indices = [item[3] for item in batch]
    questions = [item[4] for item in batch]
    choices = [item[5] for item in batch]
    return clips, labels, task_types, video_indices, questions, choices

def custom_collate_features(batch):
    features = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    task_types = [item[2] for item in batch]
    video_indices = [item[3] for item in batch]
    questions = [item[4] for item in batch]
    choices = [item[5] for item in batch]
    return features, labels, task_types, video_indices, questions, choices

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features_dir, qa_dataset):
        self.features_dir = features_dir
        self.qa_dataset = qa_dataset
    def __len__(self): return len(self.qa_dataset)
    def __getitem__(self, idx):
        item = self.qa_dataset[idx] # (vid, label, task, idx, q, choices)
        video_idx = item[3]
        if "train" in self.features_dir: local_idx = video_idx % 10000
        else: local_idx = video_idx % 5000
        feat_path = os.path.join(self.features_dir, f"feat_{local_idx:05d}.pth")
        if not os.path.exists(feat_path):
            files = os.listdir(self.features_dir)
            feat_path = os.path.join(self.features_dir, files[0]) if files else None
        features = torch.load(feat_path, map_location='cpu', weights_only=False)
        return features, item[1], item[2], item[3], item[4], item[5]

class ConsolidatedFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, consolidated_path, qa_dataset):
        print(f"Loading master features from {consolidated_path}...")
        self.features = torch.load(consolidated_path, map_location='cpu', weights_only=False)
        self.qa_dataset = qa_dataset
    def __len__(self): return len(self.qa_dataset)
    def __getitem__(self, idx):
        item = self.qa_dataset[idx]
        video_idx = item[3]
        local_idx = video_idx % self.features.shape[0]
        features = self.features[local_idx]
        return features, item[1], item[2], item[3], item[4], item[5]

def evaluate(checkpoint_path, variant='baseline', device='cuda', img_size=96, num_frames=16, quick=False, batch_size=32, suffix='', feature_dir=None):
    print(f"Starting evaluation for {variant} using checkpoint {checkpoint_path}")
    final_metrics = None
    if feature_dir:
        train_consolidated = os.path.join(feature_dir, f"{variant}_train_consolidated.pth")
        val_consolidated = os.path.join(feature_dir, f"{variant}_val_consolidated.pth")
        if os.path.exists(train_consolidated) and os.path.exists(val_consolidated):
            print(f"=> Using CONSOLIDATED master tensors from {feature_dir}")
            train_ds_qa = get_clevrer_qa_loaders(split='train', num_frames=num_frames, frame_size=img_size)
            val_ds_qa = get_clevrer_qa_loaders(split='validation', num_frames=num_frames, frame_size=img_size)
            train_ds = ConsolidatedFeatureDataset(train_consolidated, train_ds_qa)
            val_ds = ConsolidatedFeatureDataset(val_consolidated, val_ds_qa)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_features)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_features)
            probe = MultimodalChoiceProbe(192).to(device)
            probe = train_probe_on_features(probe, train_loader, val_loader, device, epochs=2)
            final_metrics = evaluate_probe_on_features(probe, val_loader, device)
        elif os.path.exists(feature_dir):
            print(f"=> Using PRE-EXTRACTED features from {feature_dir}")
            train_ds_qa = get_clevrer_qa_loaders(split='train', num_frames=num_frames, frame_size=img_size)
            val_ds_qa = get_clevrer_qa_loaders(split='validation', num_frames=num_frames, frame_size=img_size)
            train_ds = FeatureDataset(feature_dir, train_ds_qa)
            val_ds = FeatureDataset(feature_dir.replace('train', 'val'), val_ds_qa)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_features)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_features)
            probe = MultimodalChoiceProbe(192).to(device)
            probe = train_probe_on_features(probe, train_loader, val_loader, device, epochs=2)
            final_metrics = evaluate_probe_on_features(probe, val_loader, device)
    if final_metrics is None:
        params = {"img_size": img_size, "num_frames": num_frames}
        if variant == 'baseline': model = VideoJEPA(**params)
        elif variant == 'object': model = ObjectMaskedJEPA(**params)
        elif variant == 'interaction': model = InteractionAwareJEPA(**params)
        else: raise ValueError(f"Unknown variant: {variant}")
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else (checkpoint['model'] if 'model' in checkpoint else checkpoint)
            model.load_state_dict(state_dict, strict=False)
            
        model.to(device); model.eval()
        train_ds = get_clevrer_qa_loaders(split='train', num_frames=num_frames, frame_size=img_size)
        val_ds = get_clevrer_qa_loaders(split='validation', num_frames=num_frames, frame_size=img_size)
        
        # Video path MUST use num_workers=0 to prevent OOM
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)
        
        probe = train_probe(model, train_loader, val_loader, device, epochs=2)
        final_metrics = evaluate_probe(model, probe, val_loader, device)
    print(f"\nFinal Evaluation Metrics for {variant}{suffix}:", flush=True)
    for k, v in final_metrics.items(): print(f"  {k}: {v:.4f}", flush=True)
    save_dir = "/content/drive/MyDrive/object-centric-jepa/checkpoints/probes" if os.path.exists("/content/drive") else "checkpoints"
    os.makedirs(save_dir, exist_ok=True); torch.save(probe.state_dict(), os.path.join(save_dir, f"probe_{variant}{suffix}.pth"))
    print("Evaluation complete.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--variant", type=str, default="baseline")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--feature_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.variant, args.device, args.img_size, args.num_frames, False, args.batch_size, args.suffix, args.feature_dir)
