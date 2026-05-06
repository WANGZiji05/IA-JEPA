import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import json

from src.data.clevrer_dataset import CLEVRERVideoDataset
from src.models.jepa_baseline import VideoJEPA, update_target_encoder
from src.models.jepa_object import ObjectMaskedJEPA
from src.models.jepa_interaction import InteractionAwareJEPA

def parse_args():
    parser = argparse.ArgumentParser(description="Train Video JEPA Variants")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return parser.parse_args()

def save_checkpoint(state, checkpoint_dir, variant="model", epoch=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save the latest checkpoint
    last_path = os.path.join(checkpoint_dir, "last.pth")
    torch.save(state, last_path)
    
    # Save an epoch-specific checkpoint if epoch is provided
    if epoch is not None:
        # e.g., jepa_object_epoch_200.pth
        epoch_filename = f"jepa_{variant}_epoch_{epoch}.pth"
        epoch_path = os.path.join(checkpoint_dir, epoch_filename)
        torch.save(state, epoch_path)
        print(f"=> Saved checkpoint to {last_path} and {epoch_path}")
    else:
        print(f"=> Saved checkpoint to {last_path}")

def load_config(path):
    default_config = {
        "batch_size": 32,
        "accum_steps": 1,
        "num_frames": 16,
        "lr": 1.5e-4,
        "weight_decay": 0.05,
        "epochs": 100,
        "ema_momentum": 0.996,
        "mask_ratio": 0.6,
        "project_name": "Video-JEPA-CLEVRER",
        "checkpoint_dir": "checkpoints",
        "model_variant": "baseline",
        "val_interval": 1 # Validate every N epochs
    }
    try:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
            if cfg:
                default_config.update(cfg)
    except FileNotFoundError:
        print(f"Config {path} not found. Using defaults.")
    return default_config

def get_model(cfg, device):
    variant = cfg.get("model_variant", "baseline")
    params = {
        "img_size": cfg.get("frame_size", 112),
        "num_frames": cfg["num_frames"],
        "mask_ratio": cfg["mask_ratio"]
    }
    
    if variant == "baseline":
        print("Using Baseline VideoJEPA")
        return VideoJEPA(**params).to(device)
    elif variant == "motion" or variant == "object":
        print("Using ObjectMaskedJEPA (Variant A)")
        return ObjectMaskedJEPA(**params).to(device)
    elif variant == "interaction":
        print("Using InteractionAwareJEPA (Variant B)")
        return InteractionAwareJEPA(**params).to(device)
    else:
        raise ValueError(f"Unknown model variant: {variant}")

@torch.no_grad()
def validate(model, dataloader, device, cfg):
    model.eval()
    total_val_loss = 0
    for data, _ in tqdm(dataloader, desc="Validating"):
        video = data["video"].to(device)
        masks = data.get("masks").to(device) if "masks" in data else None
        collisions = data.get("collisions").to(device) if "collisions" in data else None
        
        if cfg["model_variant"] == "baseline":
            pred_latents, target_latents = model(video)
        elif cfg["model_variant"] in ["motion", "object"]:
            pred_latents, target_latents = model(video, masks=masks)
        elif cfg["model_variant"] == "interaction":
            pred_latents, target_latents = model(video, collision_frames=collisions)
            
        loss = F.mse_loss(pred_latents, target_latents)
        total_val_loss += loss.item()
        
    return total_val_loss / len(dataloader)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    variant = cfg.get("model_variant", "baseline")
    if os.path.exists("/content/drive"):
        drive_base = "/content/drive/MyDrive/object-centric-jepa/checkpoints"
        cfg["checkpoint_dir"] = os.path.join(drive_base, variant)
        print(f"=> Redirecting checkpoints to Google Drive: {cfg['checkpoint_dir']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Initialize Model
    model = get_model(cfg, device)
    
    # 2. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    
    start_epoch = 0
    global_step = 0
    history = {"train_loss": [], "val_loss": [], "epochs": []}
    
    # Try to load existing history if it exists (for plotting continuation)
    history_path = os.path.join(cfg["checkpoint_dir"], "history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            print(f"=> Loaded existing history with {len(history['epochs'])} entries")
        except Exception as e:
            print(f"=> Could not load existing history: {e}")

    # 3. Auto-resume logic
    checkpoint_path = os.path.join(cfg["checkpoint_dir"], "last.pth")
    if not os.path.exists(checkpoint_path):
        # Fallback 1: Root last.pth
        checkpoint_path = os.path.join("/content/drive/MyDrive/object-centric-jepa/checkpoints", "last.pth")
        if not os.path.exists(checkpoint_path):
             # Fallback 2: object_variant/last.pth (specific for phase 1 stretch)
             checkpoint_path = os.path.join("/content/drive/MyDrive/object-centric-jepa/checkpoints/object_variant", "last.pth")

    if (args.resume or True) and os.path.exists(checkpoint_path):
        print(f"=> Attempting to resume from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint['state_dict']

            if 'pos_embed' in state_dict:
                ckpt_pos_embed = state_dict['pos_embed']
                curr_pos_embed = model.pos_embed
                if ckpt_pos_embed.shape != curr_pos_embed.shape:
                    print(f"=> Interpolating pos_embed from {ckpt_pos_embed.shape} to {curr_pos_embed.shape}")
                    if ckpt_pos_embed.shape[1] > curr_pos_embed.shape[1]:
                        state_dict['pos_embed'] = ckpt_pos_embed[:, :curr_pos_embed.shape[1], :]
                    else:
                        new_pos_embed = curr_pos_embed.clone()
                        new_pos_embed[:, :ckpt_pos_embed.shape[1], :] = ckpt_pos_embed
                        state_dict['pos_embed'] = new_pos_embed

            model.load_state_dict(state_dict, strict=False)
            if 'optimizer' in checkpoint:
                 old_variant = checkpoint.get("config", {}).get("model_variant")
                 if variant == old_variant:
                    print(f"=> Resuming optimizer for {variant}")
                    optimizer.load_state_dict(checkpoint['optimizer'])
            
            start_epoch = checkpoint.get('epoch', 0)
            global_step = checkpoint.get('global_step', 0)
            print(f"=> Loaded checkpoint (epoch {start_epoch})")
        except Exception as e:
            print(f"=> Failed to load checkpoint: {e}")

    # 4. Initialize wandb
    wandb.init(project=cfg["project_name"], config=cfg, name=f"{variant}_{wandb.util.generate_id()}")

    # 5. Initialize Datasets and Dataloaders
    train_dataset = CLEVRERVideoDataset(
        split='train', 
        num_frames=cfg["num_frames"],
        frame_size=cfg.get("frame_size", 112),
        tensor_dir=cfg.get("tensor_dir"),
        mask_dir=cfg.get("mask_dir"),
        ann_dir=cfg.get("ann_dir")
    )
    val_dataset = CLEVRERVideoDataset(
        split='validation', 
        num_frames=cfg["num_frames"],
        frame_size=cfg.get("frame_size", 112),
        tensor_dir=cfg.get("tensor_dir") + "_val",
        mask_dir=cfg.get("mask_dir"),
        ann_dir=cfg.get("ann_dir")
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, 
                            num_workers=2, pin_memory=True, drop_last=True,
                            persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, 
                            num_workers=2, pin_memory=True, drop_last=False)

    # 6. Training Loop
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for batch_idx, (data, _) in enumerate(progress_bar):
            video = data["video"].to(device)
            masks = data.get("masks").to(device) if "masks" in data else None
            collisions = data.get("collisions").to(device) if "collisions" in data else None
            
            if cfg["model_variant"] == "baseline":
                pred_latents, target_latents = model(video)
            elif cfg["model_variant"] in ["motion", "object"]:
                pred_latents, target_latents = model(video, masks=masks)
            elif cfg["model_variant"] == "interaction":
                pred_latents, target_latents = model(video, collision_frames=collisions)
            
            loss = F.mse_loss(pred_latents, target_latents)
            loss = loss / cfg["accum_steps"]
            loss.backward()
            
            if (batch_idx + 1) % cfg["accum_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()
                update_target_encoder(model.context_encoder, model.target_encoder, momentum=cfg["ema_momentum"])
                
                wandb.log({
                    "train/loss": loss.item() * cfg["accum_steps"],
                    "train/epoch": epoch,
                    "train/step": global_step
                })
                global_step += 1
            
            epoch_train_loss += loss.item() * cfg["accum_steps"]
            progress_bar.set_postfix({'loss': loss.item() * cfg["accum_steps"]})
            
        avg_train_loss = epoch_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        if (epoch + 1) % cfg.get("val_interval", 1) == 0:
            avg_val_loss = validate(model, val_loader, device, cfg)
            print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
            wandb.log({"val/loss": avg_val_loss, "epoch": epoch + 1})
            
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["epochs"].append(epoch + 1)
            with open(os.path.join(cfg["checkpoint_dir"], "history.json"), "w") as f:
                json.dump(history, f)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'global_step': global_step,
            'config': cfg
        }, cfg["checkpoint_dir"], variant=variant, epoch=epoch + 1)

    wandb.finish()

if __name__ == "__main__":
    main()
