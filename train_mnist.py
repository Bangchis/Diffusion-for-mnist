#!/usr/bin/env python
# train_mnist.py
# ---------------------------------------------------------------------
# FP32 training for MNIST with:
# - Sparse W&B logging: loss, global/per-layer norms
# - Training speed: iterations/sec and ms/iter
# - Periodic sampling + sample speed
# - Optional FID (clean-fid) and IS (torch-fidelity)
# - Logs diffusion hyperparameters (T, beta schedule, sampling steps, eta)
# ---------------------------------------------------------------------

"""
This script implements a full training loop for a diffusion model on
MNIST.  It includes numerous metrics, sampling hooks and logging to
Weights & Biases (W&B).  In addition to the stock features, this
version encodes all visualization videos locally to MP4 via ffmpeg
before sending them to W&B.  Encoding locally avoids a bug in W&B's
automatic video conversion that sometimes misdetects channel layouts
and produces garbled green/blue stripe artifacts.  See the README or
accompanying documentation for further details.
"""

import os
import time
import math
import yaml
import torch
from torch.optim import Adam
from tqdm.auto import tqdm
from torchvision import datasets, transforms
from torchvision.utils import save_image

import wandb
import numpy as np
from torchvision.utils import make_grid

# Optional metrics: will be checked at runtime
try:
    from cleanfid import fid as clean_fid
    HAS_CLEANFID = True
except Exception:
    HAS_CLEANFID = False

try:
    from torch_fidelity import calculate_metrics as tf_calculate_metrics
    HAS_TORCH_FIDELITY = True
except Exception:
    HAS_TORCH_FIDELITY = False

from unet import UNet
from diffusion import GaussianDiffusion  # your current file name

# ---------------------------
# Utility functions for making videos
# ---------------------------


def frames_to_numpy_sequence(frames, nrow: int = 8):
    """
    Convert a list of frames (each of shape [B,C,H,W] in the range [0,1])
    into a list of H×W×3 uint8 numpy arrays suitable for video encoding.

    Each time step is first tiled into a grid using `make_grid` with
    `nrow` images per row.  Grayscale inputs are automatically
    expanded to three channels for compatibility with most video
    encoders.  This helper mirrors the logic of the original
    `frames_to_wandb_video` but returns a Python list rather than
    immediately creating a W&B Video.

    Args:
        frames: List of tensors with shape [B, C, H, W] and values in [0,1].
        nrow: Number of samples per row in the grid.

    Returns:
        List of numpy arrays with dtype uint8 and shape [H, W, 3].
    """
    seq = []
    for f in frames:
        # Clamp to [0,1] and tile into a grid of shape [C,H,W].
        f = f.clamp(0, 1)
        grid = make_grid(f, nrow=nrow)
        # Ensure we have three channels (MNIST is 1-channel).
        if grid.shape[0] == 1:
            grid = grid.repeat(3, 1, 1)
        # Convert to uint8 numpy array.
        frame = (grid * 255.0).round().byte().cpu().numpy()
        frame = np.transpose(frame, (1, 2, 0))  # [H,W,3]
        # Make contiguous to avoid stride issues when writing video.
        frame = np.ascontiguousarray(frame)
        seq.append(frame)
    return seq


def save_video_local(frames, out_path: str, fps: int = 16, nrow: int = 8):
    """
    Encode a sequence of frames into an MP4 file using ffmpeg via
    imageio.  The frames argument is a list of tensors with shape
    [B,C,H,W] in the range [0,1].  Each frame is converted to an
    H×W×3 uint8 numpy array using `frames_to_numpy_sequence`.

    Args:
        frames: List of tensors of shape [B,C,H,W] with values in [0,1].
        out_path: Destination path for the MP4 file.  Parent
            directories will be created if necessary.
        fps: Frames per second for the video.
        nrow: Number of samples per row in the tiled grid.

    Returns:
        The path to the saved video (same as `out_path`).
    """
    import imageio
    # Ensure parent directory exists.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seq = frames_to_numpy_sequence(frames, nrow=nrow)
    # Choose libx264 codec to avoid forced downscaling; set macro_block_size
    # to None so imageio does not resize frames to a multiple of 16.
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264',
                                quality=8, macro_block_size=None)
    for fr in seq:
        writer.append_data(fr)
    writer.close()
    return out_path

# ---------------------------
# Speedups on CUDA (still FP32)
# ---------------------------


def maybe_enable_cuda_speedups(cfg):
    if torch.cuda.is_available():
        if cfg.get("compute", {}).get("enable_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

# ---------------------------
# MNIST dataloader (32x32, float32)
# ---------------------------


def get_loader_mnist(bs, nw, img_size):
    tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),                          # [0,1], CxHxW
        transforms.ConvertImageDtype(torch.float32),    # force float32
    ])
    ds = datasets.MNIST(root="./data", train=True,
                        download=True, transform=tfm)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)

# ---------------------------
# Sparse norm logging helpers
# ---------------------------


def log_global_grad_norm_sparsely(model, step, every=1000):
    """
    Logs a single scalar 'train/global_grad_norm' every `every` steps.
    """
    if (step % every) != 0:
        return
    with torch.no_grad():
        norms = [p.grad.norm().item()
                 for p in model.parameters() if p.grad is not None]
        if len(norms) == 0:
            return
        global_norm = float(torch.tensor(norms).norm().item())
    wandb.log({"train/global_grad_norm": global_norm, "step": step}, step=step)

# ---------------------------
# Prepare a real-image reference folder for FID (folder-vs-folder)
# ---------------------------


def ensure_real_ref_folder(dl, out_dir, max_images=50000, img_size=32, force_rgb=False):
    """
    Exports up to `max_images` real images from the dataloader to `out_dir`
    in PNG format for FID reference.

    - MNIST is 1-channel; some FID/IS tools expect 3-channel -> set force_rgb=True to replicate channels.
    - Images are already [0,1] tensors from dataloader.
    """
    os.makedirs(out_dir, exist_ok=True)
    # If already exists with enough images, skip
    existing = [f for f in os.listdir(out_dir) if f.lower().endswith(".png")]
    if len(existing) >= max_images // 10:  # heuristic to avoid re-dumping fully
        return

    saved = 0
    idx = 0
    for x, _ in dl:
        # x: [B, C, H, W] in [0,1]
        if force_rgb and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        for i in range(x.size(0)):
            save_image(x[i], os.path.join(out_dir, f"{idx:06d}.png"))
            idx += 1
            saved += 1
            if saved >= max_images:
                return

# ---------------------------
# Generate a set of images for metrics
# ---------------------------


@torch.inference_mode()
def generate_images_to_folder(model, n_images=5000, batch_size=64, out_dir="./gen_eval", force_rgb=True):
    """
    Uses the (EMA) diffusion sampler to generate `n_images` and save as PNGs.
    Optionally tile grayscale to RGB to satisfy metric toolchains.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    idx = 0
    while saved < n_images:
        cur = min(batch_size, n_images - saved)
        imgs = model.sample(cur)  # in [0,1], shape [B, C, H, W]
        if force_rgb and imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        for i in range(cur):
            save_image(imgs[i], os.path.join(out_dir, f"{idx:06d}.png"))
            idx += 1
        saved += cur

# ---------------------------
# Compute FID (clean-fid) and IS (torch-fidelity)
# ---------------------------


def compute_fid_cleanfid(gen_dir, real_dir, device="cpu", bs=64, workers=4):
    """
    Tính FID bằng clean-fid trên CPU để tránh CUDA illegal memory access.
    """
    if not HAS_CLEANFID:
        print("[metrics] clean-fid not installed; skip FID.")
        return None
    try:
        score = clean_fid.compute_fid(
            gen_dir,              # folder ảnh sinh
            real_dir,             # folder ảnh tham chiếu
            device=device,        # <— ép "cpu"
            batch_size=bs,
            num_workers=workers
        )
        return float(score)
    except Exception as e:
        print("[metrics] clean-fid error:", e)
        return None


def compute_inception_score_torchfidelity(gen_dir, cuda=False):
    """
    Tính Inception Score bằng torch-fidelity trên CPU mặc định để an toàn.
    """
    if not HAS_TORCH_FIDELITY:
        print("[metrics] torch-fidelity not installed; skip IS.")
        return None, None
    try:
        metrics = tf_calculate_metrics(
            input1=gen_dir,
            cuda=cuda and torch.cuda.is_available(),  # mặc định False
            isc=True, fid=False, kid=False, prc=False
        )
        return float(metrics.get("inception_score_mean", float("nan"))), \
            float(metrics.get("inception_score_std",  float("nan")))
    except Exception as e:
        print("[metrics] torch-fidelity error:", e)
        return None, None

# ---------------------------
# Main training
# ---------------------------


def main(cfg_path="config_mnist_small.yaml", seed=42):
    torch.manual_seed(seed)

    # Load config and setup
    cfg = yaml.safe_load(open(cfg_path))
    os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
    os.makedirs("./samples", exist_ok=True)
    # Ensure video directory exists early to avoid race conditions
    os.makedirs("./videos", exist_ok=True)
    maybe_enable_cuda_speedups(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # W&B init
    run = None
    if cfg["wandb"]["enabled"]:
        if cfg["wandb"].get("mode", "online") == "offline":
            os.environ["WANDB_MODE"] = "offline"
        wandb.login()
        run = wandb.init(
            project=cfg["project"],
            name=cfg["run_name"],
            config=cfg,
            tags=cfg["wandb"].get("tags", [])
        )
        # Log diffusion hyperparameters once for visibility
        wandb.config.update({
            "hparams/T": cfg["diffusion"]["T"],
            "hparams/beta_schedule": cfg["diffusion"]["beta_schedule"],
            "hparams/sampling_steps": cfg["diffusion"]["sampling_steps"],
            "hparams/eta": cfg["diffusion"]["eta"],
        }, allow_val_change=True)

    # Data
    dl = get_loader_mnist(cfg["data"]["batch_size"],
                          cfg["data"]["num_workers"], cfg["data"]["image_size"])

    # Model + Diffusion (FP32 default)
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

    # Optimizer (FP32)
    opt = Adam(diffusion.parameters(),
               lr=cfg["opt"]["lr"], betas=tuple(cfg["opt"]["betas"]))

    # EMA (recommended)
    ema = None
    if cfg.get("ema", {}).get("enabled", True):
        ema = EMA(diffusion, beta=cfg["ema"]["decay"],
                  update_every=cfg["ema"]["update_every"])
        ema.to(device)

    # Train loop params
    max_steps = int(cfg["train"]["max_steps"])
    log_every = int(cfg["train"]["log_every"])
    grad_accum = int(cfg["train"].get("grad_accum", 1))

    # Norm logging params (you can add into YAML under "metrics")
    global_norm_every = int(
        cfg.get("metrics", {}).get("global_norm_every", 1000))

    # Metric config (FID / IS)
    enable_fid = bool(cfg.get("metrics", {}).get("enable_fid", False))
    enable_is = bool(cfg.get("metrics", {}).get("enable_is", False))
    fid_every = int(cfg.get("metrics", {}).get("fid_every", 4000))
    is_every = int(cfg.get("metrics", {}).get("is_every", 4000))
    metric_n_gen = int(cfg.get("metrics", {}).get("metric_num_gen", 5000))
    metric_bs = int(cfg.get("metrics", {}).get("metric_batch_size", 64))

    # Speed tracking (iterations/sec)
    step = 0
    pbar = tqdm(total=max_steps, desc="training")
    opt.zero_grad(set_to_none=True)

    # For IPS calculation over logging window
    last_log_time = time.perf_counter()
    last_log_step = 0

    # Main loop
    while step < max_steps:
        for x, _ in dl:
            # Move batch to device and force float32
            x = x.to(device, non_blocking=True).float()

            # Standard FP32 forward/backward (no AMP)
            loss = diffusion(x) / grad_accum
            loss.backward()

            if ((step + 1) % grad_accum) == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    diffusion.parameters(), cfg["opt"]["grad_clip"])
                # Optimizer update
                opt.step()
                opt.zero_grad(set_to_none=True)
                # EMA update
                if ema is not None:
                    ema.update()

            step += 1
            pbar.update(1)

            # -------- sparse scalar logging --------
            if run and step % log_every == 0:
                # training speed over last window
                now = time.perf_counter()
                delta_t = max(now - last_log_time, 1e-6)
                delta_s = step - last_log_step
                ips = delta_s / delta_t
                ms_per_iter = 1000.0 / max(ips, 1e-9)

                wandb.log({
                    "train/loss": float(loss.item() * grad_accum),
                    "speed/iter_per_sec": ips,
                    "speed/ms_per_iter": ms_per_iter,
                    "step": step
                }, step=step)

                # reset window
                last_log_time = now
                last_log_step = step

            # -------- sparse norm logging --------
            if run:
                log_global_grad_norm_sparsely(
                    diffusion, step, every=global_norm_every)

            # -------- periodic sampling (with speed) --------
            if step % int(cfg["diffusion"]["sample_every"]) == 0:
                diffusion.eval()
                with torch.inference_mode():
                    sampler = ema.ema_model if ema is not None else diffusion

                    # (a) normal sample grid + timing
                    t0 = time.perf_counter()
                    samples = sampler.sample(cfg["diffusion"]["sample_n"])
                    t1 = time.perf_counter()
                    path = f"./samples/mnist_step_{step}.png"
                    save_image(samples, path, nrow=8)
                    dt = max(t1 - t0, 1e-6)
                    imgs_per_sec = cfg["diffusion"]["sample_n"] / dt
                    if run:
                        wandb.log({
                            "samples_grid": wandb.Image(path),
                            "speed/sampling_imgs_per_sec": imgs_per_sec,
                            "speed/sampling_sec": dt,
                            "step": step
                        }, step=step)

                    # ===== NEW: two separate videos =====

                    # Common config
                    fps = int(cfg["viz"].get("video_fps", 16))

                    # ---------- (1) FORWARD VIDEO: x0 -> xT ----------
                    if cfg.get("viz", {}).get("enable_forward_traj", True) \
                       and step % int(cfg["viz"].get("forward_every_steps", 4000)) == 0:

                        Bf = int(cfg["viz"].get("forward_batch_n", 16))
                        x0_vis = x[:Bf].detach().to(device)   # [0,1]

                        # derive t-values: use list in config if provided; else build by stride
                        T = sampler.num_timesteps
                        if "forward_t_values" in cfg.get("viz", {}):
                            t_vals = list(cfg["viz"]["forward_t_values"])
                            # safety: clip to [0, T-1] and ensure T-1 is included
                            t_vals = [int(max(0, min(T-1, t))) for t in t_vals]
                            if (T-1) not in t_vals:
                                t_vals.append(T-1)
                        else:
                            stride = int(cfg["viz"].get(
                                "forward_record_every", 5))
                            t_vals = list(range(0, T, stride)) + [T-1]

                        frames_fwd = sampler.forward_noising_trajectory(
                            x0=x0_vis, t_values=t_vals)

                        # save locally (MP4)
                        fwd_path = f"./videos/forward_step_{step}.mp4"
                        save_video_local(frames_fwd, fwd_path,
                                         fps=fps, nrow=min(8, Bf))
                        if run:
                            wandb.log(
                                {"viz/forward_xt": wandb.Video(fwd_path)}, step=step)

                        # Build x_T from the SAME x0_vis for the reverse video
                        tt_T = torch.full((x0_vis.size(0),),
                                          T-1, device=device, dtype=torch.long)
                        x_T = sampler.q_sample(
                            sampler.normalize(x0_vis), tt_T)  # [-1,1]
                    else:
                        x_T = None  # may be filled below if only reverse is enabled

                    # ---------- (2) REVERSE VIDEO: xT -> x0 ----------
                    if cfg.get("viz", {}).get("enable_reverse_traj", True) \
                       and step % int(cfg["viz"].get("reverse_every_steps", 4000)) == 0:

                        Br = int(cfg["viz"].get("reverse_batch_n", 16))
                        # If we didn't run forward video this step, prepare x_T now from current batch:
                        if x_T is None:
                            x0_vis = x[:Br].detach().to(device)  # [0,1]
                            T = sampler.num_timesteps
                            tt_T = torch.full(
                                (x0_vis.size(0),), T-1, device=device, dtype=torch.long)
                            x_T = sampler.q_sample(
                                sampler.normalize(x0_vis), tt_T)  # [-1,1]

                        rec_rev = int(cfg["viz"].get(
                            "reverse_record_every", 5))
                        frames_rev = sampler.reverse_from_xt_trajectory(
                            x_t=x_T[:Br], t_start=T-1, record_every=rec_rev)

                        # save locally (MP4)
                        rev_path = f"./videos/reverse_step_{step}.mp4"
                        save_video_local(frames_rev, rev_path,
                                         fps=fps, nrow=min(8, Br))
                        if run:
                            wandb.log(
                                {"viz/reverse_xt": wandb.Video(rev_path)}, step=step)

                diffusion.train()

            # -------- sparse checkpointing --------
            if step % (5 * int(cfg["diffusion"]["sample_every"])) == 0:
                save_obj = {
                    "step": step, "model": diffusion.state_dict(), "opt": opt.state_dict()}
                if ema is not None:
                    save_obj["ema"] = ema.state_dict()
                torch.save(save_obj, os.path.join(
                    cfg["train"]["ckpt_dir"], f"mnist_step_{step}.pt"))

            # -------- optional FID & IS evaluation (thưa, tốn thời gian) --------
            # Uses folder-vs-folder: generate N images -> compare to a real-image folder we export once.
            if (enable_fid or enable_is) and (step % min(fid_every if enable_fid else is_every,
                                                         is_every if enable_is else fid_every) == 0):
                # Export real images (once) as reference
                real_ref_dir = "./metrics_ref/mnist_train_32_rgb"
                ensure_real_ref_folder(dl, real_ref_dir, max_images=50000,
                                       img_size=cfg["data"]["image_size"], force_rgb=True)

                # Generate a fresh set for metrics
                gen_dir = f"./metrics_gen/step_{step}"
                sampler = ema.ema_model if ema is not None else diffusion
                t0 = time.perf_counter()
                with torch.inference_mode():
                    generate_images_to_folder(sampler, n_images=metric_n_gen,
                                              batch_size=metric_bs, out_dir=gen_dir, force_rgb=True)
                t1 = time.perf_counter()
                gen_fps = metric_n_gen / max(t1 - t0, 1e-6)

                log_payload = {"step": step,
                               "metrics/gen_imgs_per_sec": gen_fps}

                if enable_fid and HAS_CLEANFID and (step % fid_every == 0):
                    fid_score = compute_fid_cleanfid(gen_dir, real_ref_dir)
                    if fid_score is not None:
                        log_payload["metrics/FID_clean"] = fid_score

                if enable_is and HAS_TORCH_FIDELITY and (step % is_every == 0):
                    is_mean, is_std = compute_inception_score_torchfidelity(
                        gen_dir, cuda=True)
                    if is_mean is not None:
                        log_payload["metrics/IS_mean"] = is_mean
                    if is_std is not None:
                        log_payload["metrics/IS_std"] = is_std

                if run and len(log_payload) > 1:
                    wandb.log(log_payload, step=step)

            if step >= max_steps:
                break

    pbar.close()
    if run:
        run.finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default="config_mnist_small.yaml", help="Path to YAML config")
    args = ap.parse_args()
    main(args.config)
