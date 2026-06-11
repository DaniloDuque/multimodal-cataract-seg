import torch
from torchmetrics import JaccardIndex, F1Score

from train import _forward


## @brief Evaluates a model on a dataloader and returns IoU, Dice, and F1.
#  Dice is computed manually to avoid torchmetrics version inconsistencies.
#  @param model   PyTorch model (eval mode set internally).
#  @param loader  DataLoader yielding (rgb, edge, mask).
#  @param cfg     CONFIG dict from config.py.
#  @param mode    'rgb' | 'edge' | 'early_fusion' | 'dual'.
#  @return Dict with keys 'iou', 'dice', 'f1' (scalar floats).
def compute_metrics(model, loader, cfg: dict, mode: str) -> dict:
    device = cfg["device"]
    iou_metric = JaccardIndex(task="binary").to(device)
    f1_metric  = F1Score(task="binary").to(device)

    total_dice, n = 0.0, 0

    model.eval()
    with torch.no_grad():
        for rgb, edge, mask in loader:
            rgb, edge, mask = rgb.to(device), edge.to(device), mask.to(device)
            preds = (torch.sigmoid(_forward(model, rgb, edge, mode)) > 0.5).long()
            target = mask.long()

            iou_metric.update(preds, target)
            f1_metric.update(preds.view(-1), target.view(-1))

            p, t = preds.float(), target.float()
            total_dice += (2 * (p * t).sum() / (p.sum() + t.sum() + 1e-6)).item()
            n += 1

    return {
        "iou":  iou_metric.compute().item(),
        "dice": total_dice / n,
        "f1":   f1_metric.compute().item(),
    }


## @brief Prints a formatted results table for all models.
#  @param results Dict mapping run_name -> {'iou', 'dice', 'f1'}.
def print_results_table(results: dict) -> None:
    col = 18
    header = f"{'Model':<{col}}  {'IoU':>8}  {'Dice':>8}  {'F1':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, m in results.items():
        print(f"{name:<{col}}  {m['iou']:>8.4f}  {m['dice']:>8.4f}  {m['f1']:>8.4f}")
    print("=" * len(header))
