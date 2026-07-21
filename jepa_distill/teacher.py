"""Frozen I-JEPA teacher wrappers. Interface: forward(images) -> (B, G, G, D).

T1: the official facebook/ijepa_vith14_1k via HuggingFace transformers
    (the released checkpoint IS the target encoder - what we distill from).
T2: our own ViT-B/16 pretrained in Stage 1 with the vendored official ijepa
    code (checkpoint key 'target_encoder').
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

_VENDORED_IJEPA = Path(__file__).resolve().parents[1] / "third_party" / "ijepa"
if str(_VENDORED_IJEPA) not in sys.path:
    sys.path.insert(0, str(_VENDORED_IJEPA))


class HFTeacher(nn.Module):
    """T1: HuggingFace IJepaModel. ViT-H/14 @224 -> 16x16 tokens, dim 1280."""

    def __init__(self, model_path="facebook/ijepa_vith14_1k"):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval().requires_grad_(False)
        self.embed_dim = self.model.config.hidden_size
        self.grid = self.model.config.image_size // self.model.config.patch_size

    @torch.no_grad()
    def forward(self, x):
        tokens = self.model(pixel_values=x).last_hidden_state  # (B, G*G, D), no cls token
        B, N, D = tokens.shape
        g = int(math.isqrt(N))
        assert g * g == N, f"non-square token count {N}"
        return tokens.reshape(B, g, g, D)


class IJepaCheckpointTeacher(nn.Module):
    """Target encoder from an ORIGINAL-format I-JEPA checkpoint (key
    'target_encoder'). Two uses:
      - T2: our Stage-1 run's target_encoder_final.pt
      - T1-target: Meta's released IN1K-vit.h.14-300e.pth.tar (arch=vit_huge,
        patch_size=14) - needed because the HuggingFace conversion exports the
        CONTEXT encoder ('encoder'), not the target encoder. See FLAG 5."""

    def __init__(self, checkpoint, arch="vit_base", patch_size=16, image_size=224):
        super().__init__()
        try:
            import src.models.vision_transformer as vit  # vendored ijepa repo on sys.path
        except ImportError as e:
            raise ImportError(
                f"Could not import vendored ijepa from {_VENDORED_IJEPA} - "
                "is third_party/ijepa present?") from e
        self.encoder = vit.__dict__[arch](img_size=[image_size], patch_size=patch_size)
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("target_encoder", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        assert not missing, f"missing keys loading target encoder: {missing[:5]}"
        self.encoder.eval().requires_grad_(False)
        self.embed_dim = self.encoder.embed_dim
        self.grid = image_size // patch_size

    @torch.no_grad()
    def forward(self, x):
        tokens = self.encoder(x)  # (B, N, D)
        B, N, D = tokens.shape
        g = int(math.isqrt(N))
        return tokens.reshape(B, g, g, D)


def build_teacher(kind, **kw):
    if kind == "t1_hf":
        return HFTeacher(**kw)
    if kind == "t2_ijepa":
        return IJepaCheckpointTeacher(**kw)
    raise ValueError(f"unknown teacher kind: {kind}")
