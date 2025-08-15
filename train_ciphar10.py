#!/usr/bin/env python
# train_cifar10.py
# ---------------------------------------------------------------------
# FP32 training for CIFAR-10 (RGB, 32x32) with:
# - W&B logging (loss, speed, global grad-norm)
# - Periodic sampling + sample speed
# - Two local-encoded MP4 videos (forward x0->xT, reverse xT->x0)
# - No FID/IS (disabled to keep it simple for now)
# ---------------------------------------------------------------------

import os
import time
import yaml
import torch
from torch.optim import Adam
from tqdm.auto import tqdm
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

import numpy as np
import wandb

from unet import UNet
from diffusion import GaussianDiffusion

# ---------------------------
# Video helpers (same as MNIST version)
# ---------------------------


def frames_to_numpy_sequence(frames, nrow: int = 8):
    """
    Convert list of [B,C,H,W] in [0,1] -> list of HxWx3 uint8 np arrays.
    For CIFAR-10, C=3 already, so no channel replication needed.
    """
    seq = []
    for f in frames:
        f = f.clamp(0, 1)
        grid = make_grid(f, nrow=nrow)      # [C,H,W]
        if grid.shape[0] == 1:
            # safety; CIFAR is 3ch, so usually not used
            grid = grid.repeat(3, 1, 1)
        frame = (grid * 255.0).round().byte().cpu().numpy()
        frame = np.transpose(frame, (1, 2, 0))        # [H,W,3]
        frame = np.ascontiguousarray(frame)
        seq.append(frame)
    return seq


def save_video_local(frames, out_path: str, fps: int = 16, nrow: int = 8):
    """Encode frames -> MP4 using imageio/ffmpeg, then return path."""
    import imageio
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seq = frames_to_numpy_sequence(frames, nrow=nrow)
    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264", quality=8, macro_block_size=None)
    for fr in seq:
        writer.append_data(fr)
    writer.close()
    return out_path

# ---------------------------
# CUDA speedups (TF32 + benchmark)
# ---------------------------


def maybe_enable_cuda_speedups(cfg):
    if torch.cuda.is_available():
        if cfg.get("compute", {}).get("enable_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

# ---------------------------
# CIFAR-10 dataloader (32x32 RGB, float32)
# ---------------------------


def get_loader_cifar10(bs, nw, img_size, aug=True):
    """
    CIFAR-10 training set with light aug:
    - RandomCrop(32, padding=4)
    - RandomHorizontalFlip()
    Keep outputs in [0,1] float32; diffusion module handles [-1,1].
    """
    tfms = [
        transforms.Resize(img_size),  # just in case, CIFAR-10 is already 32
    ]
    if aug:
        tfms.extend([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    tfms.extend([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
    ])
    tfm = transforms.Compose(tfms)

    ds = datasets.CIFAR10(root="./data", train=True,
                          download=True, transform=tfm)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)

# ---------------------------
# Sparse global grad-norm logging
# ---------------------------


def log_global_grad_norm_sparsely(model, step, every=1000):
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
# Main training
# ---------------------------


def main(cfg_path="config_cifar10.yaml", seed=42):
    torch.manual_seed(seed)

    # Load config & dirs
    cfg = yaml.safe_load(open(cfg_path))
    os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
    os.makedirs("./samples", exist_ok=True)
    os.makedirs("./videos", exist_ok=True)
    maybe_enable_cuda_speedups(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # W&B
    run = None
    if cfg["wandb"]["enabled"]:
        if cfg["wandb"].get("mode", "online") == "offline":
            os.environ["WANDB_MODE"] = "offline"
        wandb.login()
        run = wandb.init(
            project=cfg["project"],
            name=cfg["run_name"],
            config=cfg,
            tags=cfg["wandb"].get("tags", []),
        )
        wandb.config.update({
            "hparams/T": cfg["diffusion"]["T"],
            "hparams/beta_schedule": cfg["diffusion"]["beta_schedule"],
            "hparams/sampling_steps": cfg["diffusion"]["sampling_steps"],
            "hparams/eta": cfg["diffusion"]["eta"],
        }, allow_val_change=True)

    # Data
    dl = get_loader_cifar10(
        bs=cfg["data"]["batch_size"],
        nw=cfg["data"]["num_workers"],
        img_size=cfg["data"]["image_size"],
        aug=cfg["data"].get("augment", True),
    )

    # Model + Diffusion
    unet = UNet(
        dim=cfg["model"]["dim"],                      # e.g. 64 or 96
        dim_mults=tuple(cfg["model"]["dim_mults"]),   # e.g. [1,2,2,4]
        channels=3,                                   # CIFAR-10 RGB
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
        # T=1000 for CIFAR-10 (common)
        timesteps=cfg["diffusion"]["T"],
        # 'cosine' is fine
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        # 'pred_noise' for simplicity; try 'pred_v' later
        objective=cfg["diffusion"]["objective"],
        # =T => DDPM; <T => DDIM
        sampling_steps=cfg["diffusion"]["sampling_steps"],
        # 0.0 => deterministic DDIM
        eta=cfg["diffusion"]["eta"],
        self_condition=cfg["diffusion"]["self_condition"],
        auto_normalize=True,
        clamp_x0=cfg["diffusion"]["clamp_x0"],
    ).to(device)

    # Optimizer
    opt = Adam(diffusion.parameters(), lr=float(cfg["opt"]["lr"]), betas=tuple(
        cfg["opt"]["betas"]), weight_decay=cfg["opt"].get("weight_decay", 0.0))

    # EMA
    ema = None
    if cfg.get("ema", {}).get("enabled", True):
        ema = EMA(diffusion, beta=cfg["ema"]["decay"],
                  update_every=cfg["ema"]["update_every"])
        ema.to(device)

    # Train params
    max_steps = int(cfg["train"]["max_steps"])
    log_every = int(cfg["train"]["log_every"])
    grad_accum = int(cfg["train"].get("grad_accum", 1))
    global_norm_every = int(
        cfg.get("metrics", {}).get("global_norm_every", 1000))

    # Speed tracking
    step = 0
    pbar = tqdm(total=max_steps, desc="training")
    opt.zero_grad(set_to_none=True)
    last_log_time = time.perf_counter()
    last_log_step = 0

    while step < max_steps:
        for x, _ in dl:
            x = x.to(device, non_blocking=True).float()  # [B,3,32,32], [0,1]

            loss = diffusion(x) / grad_accum
            loss.backward()

            if ((step + 1) % grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(
                    diffusion.parameters(), cfg["opt"]["grad_clip"])
                opt.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update()

            step += 1
            pbar.update(1)

            # sparse scalars
            if run and step % log_every == 0:
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
                last_log_time = now
                last_log_step = step

            # sparse global grad-norm
            if run:
                log_global_grad_norm_sparsely(
                    diffusion, step, every=global_norm_every)

            # periodic sampling + videos
            if step % int(cfg["diffusion"]["sample_every"]) == 0:
                diffusion.eval()
                with torch.inference_mode():
                    sampler = ema.ema_model if ema is not None else diffusion

                    # samples grid
                    t0 = time.perf_counter()
                    samples = sampler.sample(
                        cfg["diffusion"]["sample_n"])  # [0,1]
                    t1 = time.perf_counter()
                    path = f"./samples/cifar10_step_{step}.png"
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

                    # videos
                    fps = int(cfg["viz"].get("video_fps", 16))

                    # forward: x0 -> xT
                    if cfg.get("viz", {}).get("enable_forward_traj", True) and \
                       step % int(cfg["viz"].get("forward_every_steps", 5000)) == 0:
                        Bf = int(cfg["viz"].get("forward_batch_n", 16))
                        x0_vis = x[:Bf].detach().to(device)   # [0,1]
                        T = sampler.num_timesteps
                        if "forward_t_values" in cfg.get("viz", {}):
                            t_vals = [int(max(0, min(T-1, t)))
                                      for t in cfg["viz"]["forward_t_values"]]
                            if (T-1) not in t_vals:
                                t_vals.append(T-1)
                        else:
                            stride = int(cfg["viz"].get(
                                "forward_record_every", 10))
                            t_vals = list(range(0, T, stride)) + [T-1]
                        frames_fwd = sampler.forward_noising_trajectory(
                            x0=x0_vis, t_values=t_vals)
                        fwd_path = f"./videos/cifar10_forward_step_{step}.mp4"
                        save_video_local(frames_fwd, fwd_path,
                                         fps=fps, nrow=min(8, Bf))
                        if run:
                            wandb.log(
                                {"viz/forward_xt": wandb.Video(fwd_path)}, step=step)

                        # build x_T for reverse
                        tt_T = torch.full((x0_vis.size(0),),
                                          T-1, device=device, dtype=torch.long)
                        x_T = sampler.q_sample(
                            sampler.normalize(x0_vis), tt_T)  # [-1,1]
                    else:
                        x_T = None

                    # reverse: xT -> x0
                    if cfg.get("viz", {}).get("enable_reverse_traj", True) and \
                       step % int(cfg["viz"].get("reverse_every_steps", 5000)) == 0:
                        Br = int(cfg["viz"].get("reverse_batch_n", 16))
                        T = sampler.num_timesteps
                        if x_T is None:
                            x0_vis = x[:Br].detach().to(device)  # [0,1]
                            tt_T = torch.full(
                                (x0_vis.size(0),), T-1, device=device, dtype=torch.long)
                            x_T = sampler.q_sample(
                                sampler.normalize(x0_vis), tt_T)
                        rec_rev = int(cfg["viz"].get(
                            "reverse_record_every", 10))
                        frames_rev = sampler.reverse_from_xt_trajectory(
                            x_t=x_T[:Br], t_start=T-1, record_every=rec_rev)
                        rev_path = f"./videos/cifar10_reverse_step_{step}.mp4"
                        save_video_local(frames_rev, rev_path,
                                         fps=fps, nrow=min(8, Br))
                        if run:
                            wandb.log(
                                {"viz/reverse_xt": wandb.Video(rev_path)}, step=step)

                diffusion.train()

            # sparse checkpoint
            if step % (5 * int(cfg["diffusion"]["sample_every"])) == 0:
                save_obj = {
                    "step": step, "model": diffusion.state_dict(), "opt": opt.state_dict()}
                if ema is not None:
                    save_obj["ema"] = ema.state_dict()
                torch.save(save_obj, os.path.join(
                    cfg["train"]["ckpt_dir"], f"cifar10_step_{step}.pt"))

            if step >= max_steps:
                break

    pbar.close()
    if run:
        run.finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default="config_cifar10.yaml", help="Path to YAML config")
    args = ap.parse_args()
    main(args.config)
