"""Image-level perturbation for the CLS-head training signal.

The CLS path needs perturbations that move the global token off-manifold *as a
whole image* — a small FG cut/paste barely shifts CLS. Combine three operators
to make CLS-perturbed views:
  1. Large CutPaste (patch covers 30-60% of the image, anywhere — not FG-restricted).
  2. Strong color jitter (brightness/contrast/saturation 0.6, hue 0.3).
  3. Optional same-batch MixUp (α=0.4).

Returns (clean_imgs, perturbed_imgs) like CutPasteUnion.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple

import torch
import torchvision.transforms as T


class GlobalCutPaste:
    """Large random-area CutPaste with no foreground mask + strong color jitter."""

    def __init__(
        self,
        area_ratio: Tuple[float, float] = (0.30, 0.60),
        aspect_ratio: float = 0.3,
        color_jitter: float = 0.6,
        hue: float = 0.3,
    ):
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.jitter = T.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter,
            hue=hue,
        )

    def _one(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        area = h * w
        target = random.uniform(*self.area_ratio) * area
        ar = random.uniform(self.aspect_ratio, 1.0 / self.aspect_ratio)
        cw = int(round(math.sqrt(target * ar)))
        ch = int(round(math.sqrt(target / ar)))
        if cw <= 0 or ch <= 0 or cw >= w or ch >= h:
            return img
        from_x = random.randint(0, w - cw)
        from_y = random.randint(0, h - ch)
        patch = img[:, from_y:from_y + ch, from_x:from_x + cw].clone()
        patch = self.jitter(patch)
        to_x = random.randint(0, w - cw)
        to_y = random.randint(0, h - ch)
        out = img.clone()
        out[:, to_y:to_y + ch, to_x:to_x + cw] = patch
        return out

    def __call__(self, imgs: torch.Tensor, subclasses: List[str] | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = imgs.clone()
        for i in range(imgs.size(0)):
            out[i] = self._one(imgs[i])
        return imgs, out


class CLSPerturb:
    """Compose GlobalCutPaste with optional within-batch MixUp."""

    def __init__(self, mixup_alpha: float = 0.0, **kwargs):
        self.cutpaste = GlobalCutPaste(**kwargs)
        self.mixup_alpha = mixup_alpha

    def __call__(self, imgs: torch.Tensor, subclasses: List[str] | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        _, perturbed = self.cutpaste(imgs)
        if self.mixup_alpha > 0 and imgs.size(0) >= 2:
            lam = float(torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample())
            perm = torch.randperm(imgs.size(0), device=imgs.device)
            perturbed = lam * perturbed + (1.0 - lam) * imgs[perm]
        return imgs, perturbed
