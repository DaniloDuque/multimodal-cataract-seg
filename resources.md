# Learning Resources

Recommended order to understand the techniques used in the paper.

## 1. U-Net (the backbone)
- **Video:** "U-Net explained" by Yannic Kilcher (YouTube)
- Covers the encoder/decoder structure and why skip connections matter. ~30 min.

## 2. Attention / Transformers
- **Video:** "Attention is All You Need explained" by Yannic Kilcher (YouTube)
  - Covers Q, K, V from scratch — the exact mechanism used in the cross-attention module.
- **Blog:** "The Illustrated Transformer" by Jay Alammar (jalammar.github.io)
  - Best visual explanation of how Q queries K/V. Read this before the video.

## 3. Cross-Attention specifically
- **Video:** "Attention in transformers, visually explained" by 3Blue1Brown (YouTube)
  - Visualizes what happens when one sequence queries another — exactly what the RGB encoder does to the edge encoder.

## 4. Medical image segmentation context
- Search "MICCAI tutorial segmentation" on YouTube for domain-specific context.

## Shortest path to understanding the paper
1. Jay Alammar's Illustrated Transformer → understand Q/K/V
2. Yannic Kilcher's U-Net video → understand encoder/decoder/skip structure
3. Re-read `paper/sections/method.tex` — it will make complete sense.
