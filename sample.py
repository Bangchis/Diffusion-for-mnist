# sample.py
# ---------------------------------------------------------------------
# Restore trained weights (EMA if available) and generate samples.
# This script mirrors the model + diffusion build in train_mnist.py,
# then loads a checkpoint and calls diffusion.sample().
# ---------------------------------------------------------------------

import os
import yaml
import torch
from torchvision.utils import save_image

from unet import UNet
from diffusion import GaussianDiffusion


def main(cfg_path="config_mnist_small.yaml",
         ckpt_path="./checkpoints/mnist_step_30000.pt",
         use_ema=True):
    """
    Load config + model, restore checkpoint, optionally load EMA weights,
    and generate a sample grid saved to 'samples_from_ckpt.png'.

    Args:
        cfg_path (str): path to the same YAML used for training.
        ckpt_path (str): path to a saved checkpoint (.pt).
        use_ema (bool): if checkpoint contains EMA state, copy EMA weights into model.
    """
    cfg = yaml.safe_load(open(cfg_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Build the same UNet ---
    unet = UNet(
        dim=cfg["model"]["dim"],
        dim_mults=tuple(cfg["model"]["dim_mults"]),
        channels=cfg["model"]["channels"],
        attn_heads=cfg["model"]["attn_heads"],
        attn_dim_head=cfg["model"]["attn_dim_head"],
        dropout=cfg["model"]["dropout"],
        self_condition=cfg["model"]["self_condition"],
        learned_variance=cfg["model"]["learned_variance"],
        outer_attn=cfg["model"]["outer_attn"],
    ).to(device)

    # --- Wrap with diffusion (same params as training) ---
    diffusion = GaussianDiffusion(
        unet,
        image_size=(cfg["data"]["image_size"], cfg["data"]["image_size"]),
        timesteps=cfg["diffusion"]["T"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        objective=cfg["diffusion"]["objective"],
        sampling_steps=cfg["diffusion"]["sampling_steps"],
        eta=cfg["diffusion"]["eta"],
        self_condition=cfg["diffusion"]["self_condition"],
        auto_normalize=True,
        clamp_x0=cfg["diffusion"]["clamp_x0"]
    ).to(device)

    # --- Load checkpoint ---
    ckpt = torch.load(ckpt_path, map_location=device)
    diffusion.load_state_dict(ckpt["model"])
    diffusion.eval()

    # --- If EMA exists and requested, copy EMA weights into 'diffusion' ---
    if use_ema and "ema" in ckpt:
        # create an EMA wrapper around current model
        ema = EMA(diffusion)
        ema.load_state_dict(ckpt["ema"])  # load EMA state from checkpoint
        ema.copy_to()                 # copy EMA weights into the wrapped model

    # --- Generate samples and save a grid image ---
    with torch.inference_mode():
        # returns [0,1] range
        imgs = diffusion.sample(cfg["diffusion"]["sample_n"])
        os.makedirs("samples", exist_ok=True)
        save_image(imgs, "samples_from_ckpt.png", nrow=8)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default="config_mnist_small.yaml", help="Path to YAML config.")
    ap.add_argument("--ckpt", type=str, default="./checkpoints/mnist_step_30000.pt",
                    help="Path to checkpoint (.pt).")
    ap.add_argument("--no-ema", action="store_true",
                    help="Disable EMA when sampling.")
    args = ap.parse_args()
    main(args.config, args.ckpt, use_ema=not args.no_ema)
