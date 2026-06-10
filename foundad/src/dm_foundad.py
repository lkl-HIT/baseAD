"""DM-FoundAD: Dual-Manifold Decoupled FoundAD.

Wraps FoundAD's VisionModule:
- Patch path (segmentation head): per-block manifold projector φ_seg^l for l ∈ L_seg.
- CLS path (image head):            per-block manifold projector φ_cls^l for l ∈ L_cls.

Both heads are FoundAD-style ViT projectors; both trained with normal-only L2
manifold-projection loss. The ONLY structural change vs FoundAD is multiple
predictors and the CLS extraction path.

Toggle DCS / DHF by config:
    L_cls = []                        -> DCS off (Idea 2, DHF-only patch pyramid)
    len(L_seg) == 1 and L_cls == [-1] -> DCS on, DHF off (Idea 1 collapsed; Pilot 2)
    len(L_seg) > 1  and L_cls == []   -> DHF on, DCS off (Pilot 3)
    len(L_seg) > 1  and len(L_cls)>=1 -> Full DM-FoundAD (Idea 1)

Layer indexing convention: integers count from the END (1 = last block).
This matches FoundAD's get_intermediate_layers(n=k) semantics — pass n=max(layers)
and slice the returned list by negative index.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModel, SiglipVisionModel

import src.dinov2.models.vision_transformer as vit
from src.utils.tensors import trunc_normal_


def _build_projector(num_patches: int, embed_dim: int, pred_emb_dim: int,
                     depth: int, if_pe: bool, feat_normed: bool) -> nn.Module:
    pred = vit.__dict__["vit_predictor"](
        num_patches=num_patches,
        embed_dim=embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=depth,
        if_pe=if_pe,
        feat_normed=feat_normed,
    )
    for m in pred.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    return pred


class ClsProjector(nn.Module):
    """Lightweight MLP projector for the CLS token (~D × pred_emb_dim × 2 params).

    A 2-layer MLP with residual: keeps φ from collapsing to identity *only*
    because we train it with image-level perturbation; without perturbation
    `φ(x) = x` is the optimum.
    """

    def __init__(self, embed_dim: int, hidden: int = 384, depth: int = 2):
        super().__init__()
        layers: List[nn.Module] = [nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, embed_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DMFoundAD(nn.Module):
    """Dual-manifold FoundAD with optional DCS / DHF.

    Args:
        model_name: "dinov3" | "dinov2" | "dino" | "clip" | "siglip"
        seg_layers: list of negative indices into get_intermediate_layers, e.g. [1,4,7,10,13]
            corresponds to UniADet's {24,21,18,15,12} on a 24-block ViT-L (1 = last block).
        cls_layers: same convention; [] disables the CLS head.
        pred_depth, pred_emb_dim: per-block patch projector hyperparams (FoundAD: 6 / 384).
        cls_hidden: hidden dim for CLS projector (default 384).
        cls_depth: depth of CLS projector (default 2).
        share_seg_backbone: if True, all seg layers share one ViT projector
            (DHF-only via score-end aggregation, cheaper).
    """

    def __init__(
        self,
        model_name: str,
        seg_layers: List[int],
        cls_layers: List[int],
        pred_depth: int = 6,
        pred_emb_dim: int = 384,
        cls_hidden: int = 384,
        cls_depth: int = 2,
        share_seg_backbone: bool = False,
        if_pe: bool = False,
        feat_normed: bool = False,
        use_cuda: bool = True,
    ):
        super().__init__()
        assert seg_layers, "seg_layers must contain at least one layer"
        self.model_name = model_name
        self.seg_layers = sorted(set(seg_layers))
        self.cls_layers = sorted(set(cls_layers))
        self.feat_normed = feat_normed
        self.share_seg_backbone = share_seg_backbone

        self.encoder, num_patches, embed_dim, self.processor = self._build_encoder(model_name)
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.max_n_layer = max(self.seg_layers + self.cls_layers + [1])

        if share_seg_backbone:
            shared = _build_projector(num_patches, embed_dim, pred_emb_dim, pred_depth, if_pe, feat_normed)
            self.seg_projectors = nn.ModuleDict({str(l): shared for l in self.seg_layers})
        else:
            self.seg_projectors = nn.ModuleDict({
                str(l): _build_projector(num_patches, embed_dim, pred_emb_dim, pred_depth, if_pe, feat_normed)
                for l in self.seg_layers
            })

        self.cls_projectors = nn.ModuleDict({
            str(l): ClsProjector(embed_dim, hidden=cls_hidden, depth=cls_depth)
            for l in self.cls_layers
        })

        self.dropout = nn.Dropout(0.2)

        if use_cuda and torch.cuda.is_available():
            self.cuda()

    # ------------------------------------------------------------------ encoder
    def _build_encoder(self, model: str):
        processor = None
        if model == "dinov2":
            enc = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval()
            num_patches, embed_dim = enc.patch_embed.num_patches, enc.embed_dim
        elif model == "dinov3":
            enc = torch.hub.load(
                "/root/.cache/torch/hub/facebookresearch_dinov3_main",
                "dinov3_vitb16", source="local",
            ).eval()
            num_patches, embed_dim = enc.patch_embed.num_patches, enc.embed_dim
        elif model == "dino":
            enc = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").eval()
            num_patches, embed_dim = 1024, enc.embed_dim
        elif model == "siglip":
            enc = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-512").eval()
            processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-512")
            num_patches, embed_dim = 1024, 768
        elif model == "clip":
            enc = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").eval()
            processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
            num_patches, embed_dim = 196, 768
        else:
            raise ValueError(f"Unknown model: {model}")
        for p in enc.parameters():
            p.requires_grad = False
        return enc, num_patches, embed_dim, processor

    # ----------------------------------------------------------------- features
    @torch.no_grad()
    def _multi_layer_features(self, imgs: torch.Tensor, paths: List[str]) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Return (patch_feats_per_layer, cls_feats_per_layer).

        Layer keys are the negative indices passed in seg_layers / cls_layers
        (1 = last block, 2 = second-to-last, ...).
        """
        patch_feats: Dict[int, torch.Tensor] = {}
        cls_feats: Dict[int, torch.Tensor] = {}

        if self.model_name in ("dinov2", "dinov3"):
            need_cls = bool(self.cls_layers)
            outs = self.encoder.get_intermediate_layers(
                imgs, n=self.max_n_layer, return_class_token=need_cls,
            )
            # outs is a list of length n; outs[-1] is the LAST block, outs[-k] is the k-th-from-last
            for l in self.seg_layers:
                feat = outs[-l]
                patch_feats[l] = feat[0] if isinstance(feat, tuple) else feat
            for l in self.cls_layers:
                feat = outs[-l]
                if isinstance(feat, tuple):
                    cls_feats[l] = feat[1]
                else:
                    cls_feats[l] = feat[:, 0, :]
        elif self.model_name == "clip":
            hs = self.encoder(pixel_values=imgs, output_hidden_states=True).hidden_states
            for l in self.seg_layers:
                patch_feats[l] = hs[-l][:, 1:, :]
            for l in self.cls_layers:
                cls_feats[l] = hs[-l][:, 0, :]
        elif self.model_name == "siglip":
            pil_list = [Image.open(p).convert("RGB") for p in paths]
            proc = self.processor(images=pil_list, return_tensors="pt")
            pv = proc["pixel_values"].to(imgs.device)
            hs = self.encoder(pixel_values=pv, output_hidden_states=True).hidden_states
            for l in self.seg_layers:
                patch_feats[l] = hs[-l]
            for l in self.cls_layers:
                # SigLIP has no CLS — use patch mean as a proxy
                cls_feats[l] = hs[-l].mean(dim=1)
        else:
            raise NotImplementedError(self.model_name)

        if self.feat_normed:
            patch_feats = {k: F.normalize(v, dim=-1) for k, v in patch_feats.items()}
            cls_feats = {k: F.normalize(v, dim=-1) for k, v in cls_feats.items()}
        return patch_feats, cls_feats

    # ---------------------------------------------------------------- predicts
    def predict_patch(self, layer: int, z: torch.Tensor) -> torch.Tensor:
        return self.seg_projectors[str(layer)](z)

    def predict_cls(self, layer: int, z: torch.Tensor) -> torch.Tensor:
        return self.cls_projectors[str(layer)](z)

    # ----------------------------------------------------------------- train
    def training_step(
        self,
        imgs_clean: torch.Tensor,
        imgs_patch_aug: torch.Tensor,
        imgs_cls_aug: torch.Tensor,
        paths: List[str],
        gate_p: float = 0.5,
        gate_c: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total normal-only manifold-projection loss.

        gate_p: probability of feeding the FG-CutPaste (patch-perturbed) image to the seg projector.
        gate_c: probability of feeding the image-level CutPaste (cls-perturbed) image to the cls projector.
        Bernoulli gating follows FoundAD's training recipe.
        """
        target_patch, target_cls = self._multi_layer_features(imgs_clean, paths)

        if torch.rand(()).item() < gate_p:
            ctx_patch, _ = self._multi_layer_features(imgs_patch_aug, paths)
        else:
            ctx_patch = target_patch

        if torch.rand(()).item() < gate_c:
            _, ctx_cls = self._multi_layer_features(imgs_cls_aug, paths)
        else:
            ctx_cls = target_cls

        loss_dict: Dict[str, torch.Tensor] = {}
        total = imgs_clean.new_zeros(())

        for l in self.seg_layers:
            z = self.dropout(ctx_patch[l])
            p = self.predict_patch(l, z)
            l_seg = F.mse_loss(p.flatten(0, 1), target_patch[l].flatten(0, 1))
            loss_dict[f"seg_l{l}"] = l_seg.detach()
            total = total + l_seg

        for l in self.cls_layers:
            z = ctx_cls[l]
            p = self.predict_cls(l, z)
            l_cls = F.mse_loss(p, target_cls[l])
            loss_dict[f"cls_l{l}"] = l_cls.detach()
            total = total + l_cls

        loss_dict["total"] = total.detach()
        return total, loss_dict

    # ----------------------------------------------------------------- score
    @torch.inference_mode()
    def score(self, imgs: torch.Tensor, paths: List[str]) -> Dict[str, torch.Tensor]:
        """Return per-image and per-pixel anomaly scores.

        Outputs:
            "patch_per_layer": {l: [B, P]} L2 distance per layer per patch
            "patch_agg":       [B, P]      uniform mean over layers
            "cls_per_layer":   {l: [B]}
            "cls_agg":         [B]         mean over cls layers (or zeros if no cls head)
            "image_topk":      [B, P]      same as patch_agg (alias for clarity)
        """
        target_patch, target_cls = self._multi_layer_features(imgs, paths)
        out: Dict[str, torch.Tensor] = {}
        per_layer_patch: Dict[int, torch.Tensor] = {}
        for l in self.seg_layers:
            z = target_patch[l]
            p = self.predict_patch(l, z)
            per_layer_patch[l] = F.mse_loss(p, z, reduction="none").mean(dim=2)
        out["patch_per_layer"] = per_layer_patch
        out["patch_agg"] = torch.stack(list(per_layer_patch.values()), 0).mean(0)

        per_layer_cls: Dict[int, torch.Tensor] = {}
        for l in self.cls_layers:
            z = target_cls[l]
            p = self.predict_cls(l, z)
            per_layer_cls[l] = F.mse_loss(p, z, reduction="none").mean(dim=1)
        out["cls_per_layer"] = per_layer_cls
        out["cls_agg"] = (
            torch.stack(list(per_layer_cls.values()), 0).mean(0)
            if per_layer_cls else imgs.new_zeros(imgs.size(0))
        )
        return out

    # ----------------------------------------------------------------- params
    def trainable_parameters(self):
        params = []
        for p in self.seg_projectors.parameters():
            if p.requires_grad:
                params.append(p)
        for p in self.cls_projectors.parameters():
            if p.requires_grad:
                params.append(p)
        return params
