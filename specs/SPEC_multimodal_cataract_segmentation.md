# SPEC: Multimodal Cataract Segmentation via Cross-Modal Attention
**Version:** 1.0  
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
- **Encoder B** processes a **derived structural map** (edge map via Canny or Sobel filter) from the same image

Both encoders run in parallel. A **cross-attention module** at the bottleneck allows Encoder A's features to query Encoder B's features — enriching the RGB representation with explicit structural/geometric information before passing to the decoder.

### Why "Multimodal"
Although both inputs come from the same image, they represent **heterogeneous feature spaces** (photometric vs. geometric), which qualifies as multimodal fusion in the architectural sense — consistent with the research topic.

### Clinical Justification
In cataract images, structural boundaries of the lens are clinically relevant for segmentation. A network that explicitly models both color and edge structure is better equipped to delineate these boundaries, especially under variable imaging conditions.

---

## 5. Architecture

```
Input Image (RGB)
       │
       ├──────────────────────────────────┐
       │                                  │
       ▼                                  ▼
  Canny/Sobel filter             Encoder A (RGB)
       │                          (U-Net or SegFormer backbone)
       ▼                                  │
  Encoder B (Edge map)                    │
  (same or lighter backbone)              │
       │                                  │
       └──────────┐          ┌────────────┘
                  ▼          ▼
            Cross-Attention Module
            Q ← Encoder A features
            K, V ← Encoder B features
                  │
                  ▼
              Decoder
                  │
                  ▼
          Segmentation Mask
```

**Backbone options (pick one):**
- U-Net (simpler, more literature, easier to modify)
- SegFormer (more modern, transformer-based, slightly more complex)

**Recommendation:** U-Net for implementation tractability; mention SegFormer as future work.

**Cross-attention:** Use PyTorch's `nn.MultiheadAttention` — do NOT implement from scratch. Connect at the bottleneck (deepest feature map level).

---

## 6. Paper Structure (IEEE LaTeX)

Write sections in this order (not the order they appear in the paper):

### Writing order:
1. **Related Work** — write first, directly from papers found
2. **Proposed Method** — describe the architecture above
3. **Experimental Design** — dataset, splits, metrics, baselines
4. **Introduction** — write after everything else; summarize motivation + contributions
5. **Abstract** — write last

### Final paper order:
1. Abstract
2. Introduction
3. Related Work
4. Proposed Method
5. Experimental Design
6. Results (if running experiments for bonus points)
7. Conclusion
8. References

---

## 7. Related Work — Three Blocks

### Block 1: Anterior Segment / Cataract Segmentation
Search terms: `cataract segmentation deep learning`, `anterior segment segmentation CNN`, `lens segmentation ophthalmic`

Key papers to find (search IEEE Xplore / Scopus):
- U-Net applied to retinal/anterior segment segmentation (2020–2025)
- nnU-Net for medical image segmentation
- Papers specifically on cataract or lens region segmentation

### Block 2: Multimodal Fusion in Medical Imaging
Search terms: `multimodal feature fusion medical image segmentation`, `cross-modal attention medical imaging`, `dual encoder segmentation`

Key papers to find:
- Cross-attention fusion architectures in medical imaging
- Early vs late vs hybrid fusion comparisons
- Multimodal ophthalmic deep learning (review papers)

Notable reference to find: *"Advances and prospects of multi-modal ophthalmic AI"* — Eye and Vision, Springer (2024)

### Block 3: SAM and Foundation Model Adaptation (for context/contrast)
Search terms: `SAM adapter medical segmentation`, `segment anything ophthalmology`, `MedSAM`

Key papers to find:
- MedSAM (Ma et al., Nature Communications, 2024)
- EyeSAM — ARVO 2024
- Learnable Ophthalmology SAM

> **Note:** This block positions your work relative to the foundation model trend, arguing that lightweight dual-encoder fusion is more practical for small datasets like the one available.

---

## 8. Experimental Design

### Dataset
- **Source:** https://universe.roboflow.com/muhammad-risma/cataract-seg/dataset/1
- **Task:** Semantic segmentation of cataract-related anterior segment structures
- **Split:** 70% train / 15% val / 15% test (or use provided split if available)

### Preprocessing
- Resize all images to 512×512
- Normalize RGB to [0,1]
- Generate edge map: `cv2.Canny(image, threshold1=50, threshold2=150)` or Sobel
- Data augmentation: horizontal flip, rotation ±15°, brightness/contrast jitter

### Baselines to Compare
| Model | Description |
|-------|-------------|
| U-Net (unimodal RGB) | Standard baseline |
| U-Net (unimodal Edge) | Edge-only input |
| U-Net (early fusion) | RGB + Edge concatenated at input |
| **Proposed (cross-attention)** | Dual encoder + cross-attention |

### Metrics
- **IoU (Intersection over Union)** — primary metric
- **Dice coefficient** — standard for medical segmentation
- **F1 score** — per class if multi-class

### Tools
- `segmentation-models-pytorch` for U-Net backbone
- `torchmetrics` for IoU and Dice
- `opencv-python` for Canny/Sobel
- `torch.nn.MultiheadAttention` for cross-attention module

---

## 9. Implementation Plan

### Files to produce
```
project/
├── data/
│   └── dataset.py          # DataLoader, augmentations, edge map generation
├── models/
│   ├── unet_baseline.py    # Standard U-Net (unimodal)
│   ├── cross_attention.py  # nn.MultiheadAttention wrapper module
│   └── dual_encoder.py     # Full proposed architecture
├── train.py                # Training loop
├── evaluate.py             # Metrics computation
└── config.py               # Hyperparameters
```

### Key implementation challenge
The cross-attention module requires reshaping feature maps from `(B, C, H, W)` to sequence format `(B, H*W, C)` before passing to `nn.MultiheadAttention`, then reshaping back. This is the main non-trivial step.

### Suggested hyperparameters
- Optimizer: AdamW, lr=1e-4
- Batch size: 8–16
- Epochs: 50–100
- Loss: BCE + Dice combined
- Cross-attention heads: 4 or 8
- Embed dim: match bottleneck channel count of U-Net (512 for standard U-Net)

---

## 10. Presentation Requirements (from assignment doc)

- Duration: 20 minutes, all members participate equally
- Font size: 18–24pt, sans-serif
- Numbered slides
- Intro slide with section outline
- Clearly distinguishable sections
- Bullets/numbering per slide
- Figures numbered with caption and source
- IEEE citation format
- Adequate color contrast

### Suggested slide structure (≈15–18 slides)
1. Title + authors
2. Outline
3. Problem motivation (why cataract segmentation matters)
4. Limitations of unimodal approaches
5. What is multimodal segmentation?
6. Related Work — Block 1 (cataract segmentation)
7. Related Work — Block 2 (multimodal fusion)
8. Related Work — Block 3 (foundation models / SAM)
9. Proposed architecture diagram
10. Cross-attention explanation
11. Dataset description
12. Experimental setup + baselines
13. Results table (or expected results if not running experiments)
14. Discussion
15. Conclusions + future work
16. References

---

## 11. Possible Citations (to verify and find full references)

> All citations must come from IEEE Xplore, Scopus, or peer-reviewed conference/journal proceedings. No arXiv preprints. Dates: 2020–2025.

| Paper | Where to find | Why cite |
|-------|--------------|----------|
| Ronneberger et al. — U-Net (2015) | MICCAI | Backbone baseline |
| Ma et al. — MedSAM (2024) | Nature Communications | Foundation model context |
| "Advances in multi-modal ophthalmic AI" — Eye and Vision (2024) | Springer | Multimodal ophthalmology survey |
| "Machine Learning for Cataract Classification/Grading" survey | Machine Intelligence Research, Springer (2022) | Identifies multimodality as future direction |
| Vaswani et al. — Attention is All You Need (2017) | NeurIPS | Cross-attention theoretical basis |
| Any cross-attention fusion paper in medical imaging (2021–2025) | IEEE TMI / MICCAI | Direct architectural precedent |
| Any anterior segment segmentation paper (2020–2025) | IEEE ISBI / MICCAI | Block 1 related work |
| EyeSAM — ARVO 2024 | IOVS journal | SAM in ophthalmology |

---

## 12. LaTeX Setup

- **Template:** IEEE conference format (`\documentclass[conference]{IEEEtran}`)
- **Recommended editor:** Overleaf (collaborative, no local setup)
- **Presentation template:** Beamer — browse https://www.overleaf.com/gallery/tagged/presentation
- **Bibliography:** BibTeX with IEEE style (`\bibliographystyle{IEEEtran}`)

---

## 13. Timeline

| Week | Task |
|------|------|
| Now | Find and read 8–12 papers for state of the art |
| +1 week | Define architecture in detail, write Related Work + Method sections |
| +2 weeks | Write Introduction + Experimental Design; start code if going for bonus |
| +3 weeks | Run experiments (bonus), write Results + Conclusion |
| Final week | Build presentation, rehearse |

---

## 14. Open Questions / Decisions Pending

- [ ] Which backbone: U-Net or SegFormer?
- [ ] Edge map method: Canny (simpler) or Sobel (differentiable, could be learned)?
- [ ] Run experiments for bonus 15 points? (Requires ~2 extra weeks of work)
- [ ] Beamer theme for presentation?
- [ ] Task distribution among 3 team members?
