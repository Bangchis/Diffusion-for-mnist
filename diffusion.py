# diffusion_core.py
# -- file này chứa các công thức toán cốt lõi của DDPM/DDIM
# -- mục tiêu: tính các hệ số từ beta-schedule, và 4 hàm quan trọng:
#    q_sample, predict_start_from_noise, predict_noise_from_start, q_posterior

# diffusion_core.py
# Core DDPM math: schedules and q/p transformations.


import math
import torch

import torch.nn as nn
import torch.nn.functional as F


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    """
    Core diffusion module that wraps a denoiser (UNet):
    - Precomputes diffusion constants (betas, alphas, etc.)
    - Provides training loss (forward): randomly pick t, add noise, regress target
    - Provides sampling loops (DDPM or DDIM)

    The denoiser must have forward(x, t, [x_self_cond]), returning a predicted target
    (epsilon, x0, or v depending on `objective`).
    """

    def __init__(self, model, *, image_size, timesteps=400, beta_schedule='cosine',
                 objective='pred_noise', sampling_steps=None, eta=0.0,
                 self_condition=False, auto_normalize=True, clamp_x0=True):
        """
        Args:
            model (nn.Module): denoiser network (e.g., UNet).
            image_size (int or (h,w)): training/sampling resolution (must match UNet).
            timesteps (int): T. Smaller (e.g., 400) is enough for MNIST.
            beta_schedule (str): only 'cosine' implemented here for simplicity.
            objective (str): 'pred_noise'|'pred_x0'|'pred_v' (training target).
            sampling_steps (int or None): if set < T => DDIM sampling with S steps; else DDPM full T.
            eta (float): DDIM stochasticity (0.0 => deterministic).
            self_condition (bool): optional self-conditioning flag.
            auto_normalize (bool): map inputs [0,1] <-> [-1,1] inside module.
            clamp_x0 (bool): clamp predicted x0 to [-1,1] during sampling for stability.
        """
        super().__init__()
        self.model = model
        param = next(model.parameters())
        param_dtype = param.dtype
        param_device = param.device
        self.channels = model.channels
        self.self_condition = self_condition
        self.objective = objective
        self.clamp_x0 = clamp_x0

        # In-module normalization helpers (kept simple & explicit)
        self.normalize = (lambda x: x * 2 -
                          1) if auto_normalize else (lambda x: x)
        self.unnormalize = (lambda x: (x + 1) *
                            0.5) if auto_normalize else (lambda x: x)

        # Normalize image_size to (H, W)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size

        # --- schedule setup ---
        if beta_schedule != 'cosine':
            raise NotImplementedError(
                "For MNIST small, keep beta_schedule='cosine'")
        betas = cosine_beta_schedule(timesteps).to(
            device=param_device, dtype=param_dtype)  # shape [T]

        alphas = 1.0 - betas                        # alpha_t
        alphas_cumprod = torch.cumprod(alphas, dim=0)        # alpha_bar_t
        alphas_cumprod_prev = F.pad(
            alphas_cumprod[:-1], (1, 0), value=1.0)  # alpha_bar_{t-1}

        # Timesteps used in training and sampling
        self.num_timesteps = int(betas.shape[0])
        self.sampling_steps = int(
            sampling_steps) if sampling_steps else self.num_timesteps
        self.is_ddim_sampling = self.sampling_steps < self.num_timesteps
        self.ddim_sampling_eta = float(eta)

        # Register constants as buffers (moved with .to(device), saved in state_dict)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # Posterior q(x_{t-1} | x_t, x_0) parameters
        posterior_variance = betas * \
            (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(
            posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas *
                             torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev)
                             * torch.sqrt(1.0 - betas) / (1.0 - alphas_cumprod))

        # Optional loss re-weighting by SNR (kept simple here)
        snr = alphas_cumprod / (1 - alphas_cumprod)
        if objective == 'pred_noise':
            loss_weight = snr / snr    # becomes 1
        elif objective == 'pred_x0':
            loss_weight = snr
        else:  # pred_v
            loss_weight = snr / (snr + 1)
        self.register_buffer('loss_weight', loss_weight)

    @property
    def device(self):
        """Convenience: returns the device where buffers live."""
        return self.betas.device

    # ----------------------
    # Forward diffusion (q)
    # ----------------------
    def q_sample(self, x0, t, noise=None):
        """
        Sample x_t from q(x_t | x_0):
            x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x0)
        return extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + \
            extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise

    # ---------------------------------
    # Converters between parameterizations
    # ---------------------------------
    def predict_start_from_noise(self, x_t, t, eps):
        """Given epsilon prediction, reconstruct x0."""
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps

    def predict_noise_from_start(self, x_t, t, x0):
        """Given x0 prediction, reconstruct epsilon."""
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x0, t, eps):
        """v-parameterization = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x0."""
        return extract(self.alphas_cumprod.sqrt(), t, x0.shape) * eps - \
            extract((1.0 - self.alphas_cumprod).sqrt(), t, x0.shape) * x0

    def predict_start_from_v(self, x_t, t, v):
        """Given v prediction, reconstruct x0."""
        return extract(self.alphas_cumprod.sqrt(), t, x_t.shape) * x_t - \
            extract((1.0 - self.alphas_cumprod).sqrt(), t, x_t.shape) * v

    # ---------------------------------
    # Model predictions at time t
    # ---------------------------------
    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False, rederive_pred_noise=False):
        """
        Run the denoiser and return (pred_noise, x0):
        - If objective == pred_noise: UNet predicts epsilon directly.
        - If objective == pred_x0:    UNet predicts x0 directly.
        - If objective == pred_v:     UNet predicts v; we convert to x0 & epsilon.

        Args:
            x (Tensor): noised image x_t.
            t (LongTensor): time indices.
            x_self_cond (Tensor|None): optional self-conditioning input.
            clip_x_start (bool): clamp x0 to [-1,1] after prediction.
            rederive_pred_noise (bool): if True, recompute epsilon from clamped x0.

        Returns:
            (pred_noise, x0) both shape like x.
        """
        out = self.model(
            x, t, x_self_cond) if x_self_cond is not None else self.model(x, t)

        maybe_clip = (lambda z: z.clamp(-1, 1)
                      ) if clip_x_start else (lambda z: z)

        if self.objective == 'pred_noise':
            pred_noise = out
            x0 = self.predict_start_from_noise(x, t, pred_noise)
            x0 = maybe_clip(x0)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x0)

        elif self.objective == 'pred_x0':
            x0 = maybe_clip(out)
            pred_noise = self.predict_noise_from_start(x, t, x0)

        else:  # 'pred_v'
            v = out
            x0 = self.predict_start_from_v(x, t, v)
            x0 = maybe_clip(x0)
            pred_noise = self.predict_noise_from_start(x, t, x0)

        return pred_noise, x0

    def q_posterior(self, x0, x_t, t):
        """
        Compute the Gaussian q(x_{t-1} | x_t, x0) parameters:
            mean = c1 * x0 + c2 * x_t
            var, log_var: closed-form from betas and alpha_bars.
        """
        mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x0 + \
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    # ----------------------
    # Training loss (forward)
    # ----------------------
    def p_losses(self, x_start, t, noise=None):
        """
        DDPM training objective:
        - Sample x_t = q(x_t | x_0)
        - Predict target according to objective and MSE it
        - (Optional) self-conditioning can be added outside for simplicity
        """
        noise = torch.randn_like(x_start) if noise is None else noise
        x = self.q_sample(x_start, t, noise)

        x_self_cond = None
        if self.self_condition and torch.rand(1, device=self.device) < 0.5:
            # simple self-conditioning: predict x0 once and feed back
            with torch.no_grad():
                _, x_self_cond = self.model_predictions(
                    x, t, None, clip_x_start=True)

        model_out = self.model(
            x, t, x_self_cond) if x_self_cond is not None else self.model(x, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:  # pred_v
            v = self.predict_v(x_start, t, noise)
            target = v

        # MSE over channels/spatial dims -> mean over batch
        loss = F.mse_loss(model_out, target, reduction='none')
        loss = loss.mean(dim=list(range(1, loss.ndim)))  # average over C,H,W
        # snr-based weight (here often ==1)
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img):
        """
        Training entry point:
        - Normalize to [-1,1]
        - Draw random timesteps
        - Compute loss
        """
        img = img.to(device=self.device, dtype=next(
            self.model.parameters()).dtype)
        b, c, h, w = img.shape
        assert (
            h, w) == self.image_size, f"image must be {self.image_size}, got {(h,w)}"
        t = torch.randint(0, self.num_timesteps, (b,),
                          device=img.device).long()
        img = self.normalize(img)
        return self.p_losses(img, t)

    # ----------------------
    # Single DDPM step p(x_{t-1}|x_t)
    # ----------------------
    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond=None):
        """
        Compute one reverse step:
            - predict (epsilon, x0), compute posterior q(x_{t-1}|x_t, x0)
            - sample from that Gaussian (add noise except at t=0)
        """
        b = x.shape[0]
        tt = torch.full((b,), t, device=self.device, dtype=torch.long)
        pred_noise, x0 = self.model_predictions(
            x, tt, x_self_cond, clip_x_start=True)
        mean, _, log_var = self.q_posterior(x0, x, tt)
        noise = torch.randn_like(x) if t > 0 else 0.0
        return mean + (0.5 * log_var).exp() * noise, x0

    # ----------------------
    # Sampling loops
    # ----------------------
    @torch.inference_mode()
    def ddpm_sample(self, shape):
        """
        DDPM sampling with T steps (slow, high quality).
        """
        img = torch.randn(shape, device=self.device)
        x0 = None
        for t in reversed(range(self.num_timesteps)):
            self_cond = x0 if self.self_condition else None
            img, x0 = self.p_sample(img, t, self_cond)
        return self.unnormalize(img)

    @torch.inference_mode()
    def ddim_sample(self, shape):
        """
        DDIM sampling with S < T steps (fast, often good quality).
        Deterministic when eta=0.0.
        """
        T, S, eta = self.num_timesteps, self.sampling_steps, self.ddim_sampling_eta
        # create a decreasing time index schedule of length S+1: [T-1, ..., 0, -1]
        times = torch.linspace(-1, T - 1, steps=S + 1,
                               device=self.device).long().flip(0)
        pairs = list(zip(times[:-1].tolist(), times[1:].tolist()))

        img = torch.randn(shape, device=self.device)
        x0 = None

        for t, t_next in pairs:
            tt = torch.full(
                (shape[0],), t, device=self.device, dtype=torch.long)
            pred_noise, x0 = self.model_predictions(
                img, tt, None, clip_x_start=True, rederive_pred_noise=True)

            if t_next < 0:
                # final step: directly set to predicted x0
                img = x0
                continue

            a_t, a_next = self.alphas_cumprod[t], self.alphas_cumprod[t_next]
            sigma = eta * ((1 - a_t / a_next) *
                           (1 - a_next) / (1 - a_t)).sqrt()
            c = (1 - a_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)

            # DDIM update rule
            img = x0 * a_next.sqrt() + c * pred_noise + sigma * noise

        return self.unnormalize(img)

    @torch.inference_mode()
    def sample(self, batch_size=16):
        """
        Public sampling API:
            - choose DDPM or DDIM depending on `sampling_steps`
            - returns a batch of images in [0,1]
        """
        H, W = self.image_size
        fn = self.ddim_sample if self.is_ddim_sampling else self.ddpm_sample
        return fn((batch_size, self.channels, H, W))

    # In diffusion_core.py (add these methods inside GaussianDiffusion)

    # ----------------------
    # DDPM sampling with trajectory recording and foward transformations
    # ----------------------

    @torch.inference_mode()
    def ddpm_sample_trajectory(self, shape, record_every=50, return_x0=False):
        """
        DDPM sampling but also record intermediate frames.
        - record_every: save a snapshot every N steps (also includes first/last).
        - return_x0: if True, also store predicted x0 at the same checkpoints.

        Returns:
            final_img [B,C,H,W] in [0,1],
            frames_xt: list of tensors in [0,1], each [B,C,H,W]
            frames_x0 (or None): same length as frames_xt if return_x0=True
        """
        img = torch.randn(shape, device=self.device)
        frames_xt = []
        frames_x0 = [] if return_x0 else None

        x0 = None
        T = self.num_timesteps

        for t in reversed(range(T)):
            # record current x_t before stepping
            if t == T - 1 or t == 0 or (t % record_every) == 0:
                # unnormalize for visualization (to [0,1])
                frames_xt.append(self.unnormalize(img.clamp(-1, 1)))
                if return_x0 and x0 is not None:
                    frames_x0.append(self.unnormalize(x0.clamp(-1, 1)))

            self_cond = x0 if self.self_condition else None
            img, x0 = self.p_sample(img, t, self_cond)

        # record the final image
        frames_xt.append(self.unnormalize(img.clamp(-1, 1)))
        if return_x0:
            frames_x0.append(self.unnormalize(x0.clamp(-1, 1)))

        return self.unnormalize(img), frames_xt, frames_x0

    @torch.no_grad()
    def forward_noising_trajectory(self, x0, t_values, *, return_xt_last=False):
        """
        Create a forward (noising) trajectory for visualization: x0 -> ... -> x_T.

        Args:
            x0 (Tensor): clean images in [0,1], shape [B,C,H,W] (from DataLoader).
            t_values (Iterable[int]): list of discrete times (0..T-1) you want to record.
                                    Example: [0, 10, 20, 40, 80, 160, 320, 399]
            return_xt_last (bool): if True, also returns the exact x_T used for the
                                last time in t_values (in [-1,1]), so you can feed
                                it to reverse_from_xt_trajectory for a perfectly
                                matched reverse video.

        Returns:
            frames (List[Tensor]): each element is [B,C,H,W] in [0,1]
            (optional) xt_last (Tensor): the last noisy batch in [-1,1], shape [B,C,H,W]
                                        ONLY if return_xt_last=True
        """
        # ---- 1) Validate inputs ----
        assert x0.ndim == 4, f"x0 must be [B,C,H,W], got {x0.shape}"
        B, C, H, W = x0.shape
        assert C == self.channels, f"channel mismatch: x0 has {C}, model has {self.channels}"
        T = self.num_timesteps

        # sanitize t_values: ints, clipped to [0, T-1], unique & sorted (increasing noise)
        t_list = sorted({int(max(0, min(T - 1, int(t)))) for t in t_values})
        if len(t_list) == 0:
            return [] if not return_xt_last else ([], None)

        # ---- 2) Move & normalize once ----
        x0 = x0.to(device=self.device, dtype=next(
            self.model.parameters()).dtype)
        x0n = self.normalize(x0)  # [0,1] -> [-1,1]

        # ---- 3) Use ONE fixed epsilon for the whole batch so the path is smooth ----
        # This makes frames for different t lie on the same noise direction,
        # i.e., xt = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*epsilon
        eps = torch.randn_like(x0n)

        frames = []
        xt_last = None

        # ---- 4) Produce frames at requested times ----
        for t in t_list:
            tt = torch.full((B,), t, device=self.device, dtype=torch.long)

            # equivalent to: xt = sqrt(alpha_bar_t)*x0n + sqrt(1-alpha_bar_t)*eps
            # using the SAME eps for all t (smooth trajectory)
            xt = (
                self.sqrt_alphas_cumprod.index_select(0, tt)
                .reshape(B, 1, 1, 1) * x0n
                +
                self.sqrt_one_minus_alphas_cumprod.index_select(0, tt)
                .reshape(B, 1, 1, 1) * eps
            )

            # store a viewable frame in [0,1]
            frames.append(self.unnormalize(xt.clamp(-1, 1)))

            xt_last = xt  # keep the last noisy batch we produced

        # ---- 5) Return frames (and optionally the exact last xt in [-1,1]) ----
        if return_xt_last:
            return frames, xt_last
        return frames

    @torch.inference_mode()
    def reverse_from_xt_trajectory(self, x_t, t_start=None, record_every=50):
        if t_start is None:
            t_start = self.num_timesteps - 1

        img = x_t.clone()            # x_t ở miền [-1,1]
        frames = []
        x0 = None

        for t in reversed(range(t_start + 1)):  # t_start, ..., 0
            # Ghi lại x_t hiện tại (chưa bước) để thấy diễn tiến mượt
            if t == t_start or t == 0 or (t % record_every) == 0:
                frames.append(self.unnormalize(
                    img.clamp(-1, 1)))  # [0,1] để xem

            self_cond = x0 if self.self_condition else None
            img, x0 = self.p_sample(img, t, self_cond)  # 1 bước DDPM

        # Đảm bảo khung cuối là x_0
        frames.append(self.unnormalize(img.clamp(-1, 1)))
        return frames
