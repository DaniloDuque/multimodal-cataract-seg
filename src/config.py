import torch

CONFIG = {
    "img_size":        512,
    "batch_size":      8,
    "epochs":          100,
    "lr":              1e-4,
    "weight_decay":    1e-2,
    "canny_t1":        50,
    "canny_t2":        150,
    "n_heads":         8,
    "embed_dim":       512,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "data_root":       "data/cataract-seg",
    "checkpoints_dir": "checkpoints/",
    "figures_dir":     "../figures",
    "seed":            42,
}
