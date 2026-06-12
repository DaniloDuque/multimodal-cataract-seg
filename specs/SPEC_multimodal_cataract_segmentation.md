# SPEC: Multimodal Cataract Segmentation via Cross-Modal Attention
**Version:** 2.0
**Course:** Neural Networks — PARMA Group, Instituto Tecnológico de Costa Rica
**Assignment:** Investigación Corta (short research paper + presentation)
**Due:** Week 17 (Wednesday)
**Team size:** 3 people
**Language:** English

---

## 1. Context & Assignment Requirements

- **Deliverable 1 (70pts):** IEEE-format research paper in LaTeX, max 8 pages
- **Deliverable 2 (30pts):** 20-minute presentation (Beamer/LaTeX), all members participate equally
- **Bonus (15pts):** Actually run the experiments and report real results
- **Submission:** .zip with LaTeX source + PDF via TEC-Digital
- **Dataset:** https://universe.roboflow.com/muhammad-risma/cataract-seg/dataset/1
- **State of the art constraint:** Papers no older than 5 years, no arXiv preprints — only conference/journal publications (IEEE Xplore, Scopus, MICCAI, etc.)

---

## 2. Research Topic

**Topic 2 — Multimodal Segmentation Models** applied to ophthalmic (cataract) images.

---

## 3. Title

> **"Cross-Modal Attention Fusion for Anterior Segment Segmentation Using Complementary Image Representations"**

---

## 4. Core Idea / Approach

### Problem
Unimodal segmentation of anterior segment structures (cornea, pupil, lens) in cataract images is sensitive to lighting conditions and artifacts inherent to a single image representation. The opacified lens sometimes has low color contrast but detectable geometric/structural edges.

### Proposed Solution
A dual-encoder architecture where:
- **Encoder A** processes the original **RGB image** (color + texture features)
- **Encoder B** processes a **derived structural map** (Canny edge map) from the same image

Both encoders run in parallel. A **cross-attention module** at the bottleneck allows Encoder A's features to query Encoder B's features — enriching the RGB representation with explicit structural/geometric information before passing to the decoder.

### Why "Multimodal"
Although both inputs come from the same image, they represent **heterogeneous feature spaces** (photometric vs. geometric), which qualifies as multimodal fusion in the architectural sense.

### Backbone Decision
**U-Net** — chosen for implementation tractability and literature support. SegFormer is left as future work.

### Edge Map Decision
**Canny** with t1=50, t2=150 — chosen for simplicity. Learnable Sobel is left as future work.

---

## 5. Architecture

```
Input Image (RGB)
       │
       ├──────────────────────────────────┐
       │                                  │
       ▼                                  ▼
  Canny filter (t1=50, t2=150)     Encoder A (U-Net, RGB)
       │                            64→64→128→256→512 channels
       ▼                                  │
  Encoder B (U-Net, Edge map)             │
  (same architecture, no shared weights)  │
       │                                  │
       ├─────────── feats[1..4] ──────────┤  ← edge skips fused via 1×1 proj
       │                                  │
       │         feats[4] (256ch)         │
       └──────────┐          ┌────────────┘
                  ▼          ▼
         Cross-Attention Module (stage 4, 256ch)   ← NEW
         Q ← F_A[4]  K, V ← F_B[4]
         n_heads = 8, gated
                  │
                  ▼
         feats[5] (512ch, bottleneck)
       ┌──────────┘          └────────────┐
       ▼                                  ▼
  F_B (edge)                         F_A (RGB)
       └──────────┐          ┌────────────┘
                  ▼          ▼
         Cross-Attention Module (bottleneck, 512ch)
         Q ← F_A  K, V ← F_B
         n_heads = 8, gated scalar gate
                  │
                  ▼
     Decoder (expanding path)
       skip at each stage = proj(cat(edge_skip, rgb_skip))  ← NEW
                  │
                  ▼
         Segmentation Mask  (sigmoid output)
```

**Three architectural improvements over v1:**

1. **Learnable scalar gate** in `CrossAttentionFusion`: `gate = sigmoid(θ)` multiplies the attention output before the residual add. Initialized to `θ = -4` so the edge contribution starts near-zero and grows only if it helps.

2. **Second cross-attention at encoder stage 4** (256ch, 32×32 spatial): injects structural information one level above the bottleneck for richer multi-scale context.

3. **Edge encoder skip connections**: at each decoder stage, the edge encoder's feature map at the matching resolution is channel-concatenated with the RGB skip connection and projected back to the original channel count via a 1×1 conv.

**Key implementation note:** Reshape feature maps from `(B, C, H, W)` → `(B, H*W, C)` before `nn.MultiheadAttention`, reshape back after.

---

## 6. Paper Structure (IEEE LaTeX)

### Status

| Section | File | Status |
|---------|------|--------|
| Abstract | `main.tex` | ✅ Written |
| Introduction | `sections/introduction.tex` | ⬜ TODO |
| Related Work | `sections/related_work.tex` | ✅ Written |
| Proposed Method | `sections/method.tex` | ✅ Written |
| Experimental Design | `sections/experimental_design.tex` | ✅ Written |
| Results | `sections/results.tex` | ⬜ Requires experiments |
| Conclusion | `sections/conclusion.tex` | ⬜ TODO |

### Final paper order
1. Abstract
2. Introduction
3. Related Work
4. Proposed Method
5. Experimental Design
6. Results (bonus)
7. Conclusion
8. References

---

## 7. References in Use

> All verified peer-reviewed. No arXiv preprints. DOIs confirmed.

| Key | Paper | Venue | Year |
|-----|-------|-------|------|
| `cataractseg2023roboflow` | Cataract-Seg Dataset — Muhammad Risma | Roboflow Universe | 2023 |
| `ronneberger2015unet` | U-Net — Ronneberger et al. | MICCAI | 2015 |
| `grammatikopoulou2021cadis` | CaDIS — Grammatikopoulou et al. | Medical Image Analysis | 2021 |
| `vaswani2017attention` | Attention Is All You Need — Vaswani et al. | NeurIPS | 2017 |
| `zhang2022mmformer` | mmFormer — Zhang et al. | MICCAI | 2022 |
| `li2024dectnet` | DECTNet — Li et al. | PLOS ONE | 2024 |
| `wang2024multimodal` | Multi-modal ophthalmic AI survey — Wang et al. | Eye and Vision | 2024 |
| `ma2024medsam` | MedSAM — Ma et al. | Nature Communications | 2024 |
| `chen2023samadapter` | SAM-Adapter — Chen et al. | ICCV Workshop | 2023 |

### Removed references (and why)
| Removed | Reason |
|---------|--------|
| nnU-Net (Isensee 2021) | Not a baseline in experiments; no distinct argument |
| DeepPyram (Ghamsarian 2022) | Motivation link to proposed work was a stretch |
| DATTNet (Zhang 2024) | Different concept (dual attention heads ≠ dual encoders); redundant with DECTNet |
| RETFound (Zhou 2023) | Wrong task (detection) and wrong anatomy (fundus) |
| VisionFM (Qiu 2024) | Redundant with MedSAM for the same argument |

---

## 8. Experimental Design

### Dataset
- **Source:** Roboflow Cataract-Seg (`cataractseg2023roboflow`)
- **Split:** 70% train / 15% val / 15% test (stratified)

### Preprocessing
- Resize to 512×512
- Normalize RGB to [0,1]
- Edge map: `cv2.Canny(gray, 50, 150)`, replicated to 3 channels

### Augmentation (train only)
- Horizontal flip (p=0.5)
- Rotation ±15°
- Brightness/contrast jitter (factor [0.8, 1.2])
- Edge map recomputed after geometric transforms

### Baselines
| Model | Description |
|-------|-------------|
| U-Net (RGB) | Single encoder, RGB only |
| U-Net (Edge) | Single encoder, edge map only |
| U-Net (Early Fusion) | Single encoder, RGB+Edge concatenated at input |
| **Proposed** | Dual encoder + bottleneck cross-attention |

### Metrics
- **IoU** — primary
- **Dice coefficient**
- **F1** (per class if multi-class)

### Hyperparameters
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-2
- Batch size: 8
- Epochs: 100
- Loss: BCE + Dice
- Cross-attention heads: 8
- Bottleneck embed dim: 512

---

## 9. Implementation Plan

### Project structure
```
project/
├── experiment.ipynb        # Orchestration notebook (runs all steps in order)
├── data/
│   └── dataset.py          # CataractSegDataset, augmentations, edge map generation
├── models/
│   ├── unet_baseline.py    # Standard U-Net (unimodal, works for RGB, Edge, Early Fusion)
│   ├── cross_attention.py  # CrossAttentionFusion module (reshape → MHA → reshape)
│   └── dual_encoder.py     # Full proposed model (calls unet_baseline + cross_attention)
├── train.py                # train_one_epoch(), validate() functions
├── evaluate.py             # compute_metrics() — IoU, Dice, F1 via torchmetrics
└── config.py               # Single CONFIG dict with all hyperparameters
```

### experiment.ipynb — cell structure
```
[1] Install / import dependencies
[2] Load config (from config.py)
[3] Build datasets and dataloaders (from data/dataset.py)
[4] Train baseline: U-Net RGB        → save checkpoint
[5] Train baseline: U-Net Edge       → save checkpoint
[6] Train baseline: U-Net Early Fusion → save checkpoint
[7] Train proposed: Dual Encoder     → save checkpoint
[8] Evaluate all 4 models on test set → metrics table
[9] Plot: loss curves, IoU curves, sample predictions side-by-side
[10] Print final results table (IoU / Dice / F1 per model)
```

### Key implementation notes

**dataset.py**
- `CataractSegDataset.__getitem__` returns `(rgb, edge, mask)` always; baselines just ignore the edge or concatenate it
- Edge map recomputed after geometric augmentations, not stored on disk

**unet_baseline.py**
- Accepts `in_channels` parameter: 3 for RGB/Edge, 6 for Early Fusion
- Returns `(bottleneck_features, skip_connections, logits)` so `dual_encoder.py` can reuse the encoder

**cross_attention.py**
```python
# Core logic (used at both bottleneck and stage 4):
# x_rgb: (B, C, H, W)  →  (B, H*W, C)  →  Q
# x_edge: (B, C, H, W) →  (B, H*W, C)  →  K, V
# gate = sigmoid(θ),  θ initialized to -4  → starts near-zero
# out = gate * MultiheadAttention(Q, K, V) + Q  (gated residual)
# out: (B, H*W, C)  →  (B, C, H, W)
# Two instances: CrossAttentionFusion(512, 8) and CrossAttentionFusion(256, 8)
```

**dual_encoder.py — forward pass summary**
```python
feats_rgb  = encoder_A(rgb)   # list[6]: channels [3,64,64,128,256,512]
feats_edge = encoder_B(edge)

# Stage-4 cross-attention (256ch)
feats_rgb[4] = fusion_stage4(feats_rgb[4], feats_edge[4])

# Bottleneck cross-attention (512ch)
feats_rgb[5] = fusion_bottleneck(feats_rgb[5], feats_edge[5])

# Edge skip fusion at each resolution level (feats[1..4])
for i, proj in enumerate(skip_projs):
    feats_rgb[i+1] = proj(cat([feats_rgb[i+1], feats_edge[i+1]], dim=1))

out = decoder(feats_rgb)
```

**config.py**
```python
CONFIG = {
    "img_size": 512,
    "batch_size": 8,
    "epochs": 100,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "canny_t1": 50,
    "canny_t2": 150,
    "n_heads": 8,
    "embed_dim": 512,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_root": "data/cataract-seg",
    "checkpoints_dir": "checkpoints/",
}
```

### Dependencies
```
torch
segmentation-models-pytorch
torchmetrics
opencv-python
albumentations
matplotlib
```

---

## 10. Presentation Requirements

- Duration: 20 minutes, all members participate equally
- Font size: 18–24pt, sans-serif
- Numbered slides, IEEE citation format

### Suggested slide structure (≈15–18 slides)
1. Title + authors
2. Outline
3. Problem motivation
4. Limitations of unimodal approaches
5. What is multimodal segmentation?
6. Related Work — Block 1 (cataract segmentation)
7. Related Work — Block 2 (multimodal fusion)
8. Related Work — Block 3 (foundation models / SAM)
9. Proposed architecture diagram
10. Cross-attention explanation
11. Dataset description
12. Experimental setup + baselines
13. Results table
14. Discussion
15. Conclusions + future work
16. References

---

## 11. LaTeX Setup

- **Template:** `\documentclass[conference]{IEEEtran}`
- **Build:** `pdflatex → bibtex → pdflatex → pdflatex`, output to `paper/out/`
- **Bibliography:** BibTeX, `\bibliographystyle{IEEEtran}`, source in `paper/refs.bib`
- **Extra package added:** `\usepackage{amssymb}` (required for `\mathbb{R}`)

---

## 12. Open Questions / Decisions Pending

- [ ] Run experiments for bonus 15 points? (requires GPU + ~2 weeks)
- [ ] Beamer theme for presentation?
- [ ] Task distribution among 3 team members?
