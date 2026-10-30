"""Text-image verifiers used to select among candidates (protocol: V1, V2).

Both are OpenCLIP models with remote-sensing weights, loaded from pinned Hub revisions.
A verifier returns the cosine similarity between each image and its caption.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

VERIFIERS = {
    "v1_remoteclip_b32": dict(arch="ViT-B-32", repo="chendelong/RemoteCLIP",
                              file="RemoteCLIP-ViT-B-32.pt", revision="bf1d8a3c"),
    "v2_georsclip_l14": dict(arch="ViT-L-14", repo="Zilun/GeoRSCLIP",
                             file="ckpt/RS5M_ViT-L-14.pt", revision="4920188e"),
}


def _clean_state_dict(sd):
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}


@dataclass
class ClipVerifier:
    name: str
    model: torch.nn.Module
    preprocess: object
    tokenizer: object
    device: str

    @classmethod
    def load(cls, name: str, device: str):
        import open_clip
        from huggingface_hub import hf_hub_download
        spec = VERIFIERS[name]
        path = hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"])
        model, _, preprocess = open_clip.create_model_and_transforms(spec["arch"], pretrained=None)
        sd = _clean_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        res = model.load_state_dict(sd, strict=False)
        if res.missing_keys:
            raise RuntimeError(f"{name}: {len(res.missing_keys)} missing keys, first {res.missing_keys[:5]}")
        if res.unexpected_keys:
            print(f"[verifier] {name}: ignoring {len(res.unexpected_keys)} unexpected keys")
        model = model.to(device).eval()
        return cls(name, model, preprocess, open_clip.get_tokenizer(spec["arch"]), device)

    @torch.no_grad()
    def text_features(self, captions: list[str]) -> torch.Tensor:
        tok = self.tokenizer(captions).to(self.device)     # truncates to the 77-token context
        return torch.nn.functional.normalize(self.model.encode_text(tok).float(), dim=-1)

    @torch.no_grad()
    def image_features(self, images) -> torch.Tensor:
        x = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        return torch.nn.functional.normalize(self.model.encode_image(x).float(), dim=-1)

    @torch.no_grad()
    def score(self, images, captions: list[str]) -> torch.Tensor:
        """Cosine similarity of each image with its own caption."""
        return (self.image_features(images) * self.text_features(captions)).sum(-1).cpu()
