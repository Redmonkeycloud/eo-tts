"""Evaluators of the protocol (E1–E5). Never used to select samples.

E1–E3 reproduce the upstream TerraDiT metric code per image (same models, same input
handling) so that their means equal what terradit/eval/metrics.py reports. E4 and E5 return
features; KID is computed from them in eotts.stats.kid.
"""
from __future__ import annotations

import numpy as np
import torch

CLIP_EVAL_ID = "openai/clip-vit-large-patch14"
CLIP_EVAL_REVISION = "32bd6428"
DINO_ID = "facebook/dinov2-large"
DINO_REVISION = "47b73eef"


def to_u8(img) -> torch.Tensor:
    return torch.from_numpy(np.array(img.convert("RGB"), dtype=np.uint8)).permute(2, 0, 1)


class ClipEval:
    """E1: cosine of OpenAI CLIP ViT-L/14 image and caption embeddings (upstream compute_clip / 100)."""

    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(CLIP_EVAL_ID, revision=CLIP_EVAL_REVISION).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_EVAL_ID, revision=CLIP_EVAL_REVISION)
        self.device = device

    @torch.no_grad()
    def __call__(self, images, captions) -> np.ndarray:
        inp = self.proc(text=list(captions), images=[i.convert("RGB") for i in images],
                        return_tensors="pt", padding=True, truncation=True)
        out = self.model(**{k: v.to(self.device) for k, v in inp.items()})
        logits = torch.diagonal(out.logits_per_image).float()
        return (logits / self.model.logit_scale.exp().float()).cpu().numpy()


class PairwiseEval:
    """E2 LPIPS (SqueezeNet, inputs in [−1, 1]) and E3 SSIM (data range 1) against ground truth."""

    def __init__(self, device):
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, images, gts) -> tuple[np.ndarray, np.ndarray]:
        from PIL import Image
        size = gts[0].size
        gen = torch.stack([to_u8(i.convert("RGB").resize(size, Image.BILINEAR) if i.size != size else i)
                           for i in images]).to(self.device).float() / 255
        gt = torch.stack([to_u8(g) for g in gts]).to(self.device).float() / 255
        lp = np.array([float(self.lpips(2 * g[None] - 1, 2 * t[None] - 1)) for g, t in zip(gen, gt)])
        ss = self.ssim(gen, gt).float().cpu().numpy()
        return lp, ss


class InceptionFeatures:
    """E4 features: torchmetrics' Inception-v3 (the FID/KID network), 2048-d pool features."""

    def __init__(self, device):
        from torchmetrics.image.fid import NoTrainInceptionV3
        self.net = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, images) -> np.ndarray:
        x = torch.stack([to_u8(i) for i in images]).to(self.device)
        return self.net(x).float().cpu().numpy()


class DinoFeatures:
    """E5 features: DINOv2 ViT-L/14 CLS token after the final layer norm (1024-d).

    Images are resized to 224×224 (bicubic) without cropping and normalised with ImageNet
    statistics, as in Stein et al. (2023), so the whole 256×256 tile is seen.
    """

    def __init__(self, device):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(DINO_ID, revision=DINO_REVISION).to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.device = device

    @torch.no_grad()
    def __call__(self, images) -> np.ndarray:
        from PIL import Image
        x = torch.stack([to_u8(i.convert("RGB").resize((224, 224), Image.BICUBIC)) for i in images])
        x = (x.to(self.device).float() / 255 - self.mean) / self.std
        return self.model(pixel_values=x).pooler_output.float().cpu().numpy()
