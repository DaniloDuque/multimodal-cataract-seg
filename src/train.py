import os
import torch
import torch.nn as nn


## @brief Combined BCE + Dice loss for binary segmentation.
#  @param pred   Logits (B, 1, H, W).
#  @param target Binary mask (B, 1, H, W).
#  @return Scalar loss.
def segmentation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce  = nn.functional.binary_cross_entropy_with_logits(pred, target)
    prob = torch.sigmoid(pred)
    dice = 1.0 - (2 * (prob * target).sum()) / (prob.sum() + target.sum() + 1e-6)
    return bce + dice


## @brief Runs one training epoch.
#  @param model      PyTorch model. Receives (rgb, edge) or just rgb depending on mode.
#  @param loader     Training DataLoader yielding (rgb, edge, mask).
#  @param optimizer  Optimizer.
#  @param device     torch.device.
#  @param mode       'rgb' | 'edge' | 'early_fusion' | 'dual'
#  @return Mean loss over the epoch.
def train_one_epoch(model, loader, optimizer, device: str, mode: str) -> float:
    model.train()
    total_loss = 0.0
    for rgb, edge, mask in loader:
        rgb, edge, mask = rgb.to(device), edge.to(device), mask.to(device)
        optimizer.zero_grad()
        pred = _forward(model, rgb, edge, mode)
        loss = segmentation_loss(pred, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


## @brief Runs one validation epoch (no gradients).
#  @return Mean loss over the epoch.
def validate(model, loader, device: str, mode: str) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for rgb, edge, mask in loader:
            rgb, edge, mask = rgb.to(device), edge.to(device), mask.to(device)
            pred = _forward(model, rgb, edge, mode)
            total_loss += segmentation_loss(pred, mask).item()
    return total_loss / len(loader)


## @brief Full training loop with checkpoint saving.
#  @param model       PyTorch model.
#  @param train_loader Training DataLoader.
#  @param val_loader   Validation DataLoader.
#  @param cfg          CONFIG dict from config.py.
#  @param mode         Input mode string.
#  @param run_name     Name used for the saved checkpoint file.
#  @return Dict with 'train_losses' and 'val_losses' lists.
def train(model, train_loader, val_loader, cfg: dict, mode: str, run_name: str) -> dict:
    device    = cfg["device"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    os.makedirs(cfg["checkpoints_dir"], exist_ok=True)

    history = {"train_losses": [], "val_losses": []}
    best_val = float("inf")

    for epoch in range(cfg["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, mode)
        val_loss   = validate(model, val_loader, device, mode)

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       os.path.join(cfg["checkpoints_dir"], f"{run_name}_best.pth"))

        if (epoch + 1) % 10 == 0:
            print(f"[{run_name}] Epoch {epoch+1}/{cfg['epochs']}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")

    return history


# ── Internal helper ──────────────────────────────────────────────────────────

def _forward(model, rgb, edge, mode: str) -> torch.Tensor:
    if mode == "rgb":
        return model(rgb)
    if mode == "edge":
        return model(edge)
    if mode == "early_fusion":
        return model(torch.cat([rgb, edge], dim=1))
    if mode == "dual":
        return model(rgb, edge)
    raise ValueError(f"Unknown mode: {mode}")
