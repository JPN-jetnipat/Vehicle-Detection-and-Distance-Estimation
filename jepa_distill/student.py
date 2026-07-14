"""YOLOv5 backbone student: build from yolov5s.yaml, tap P4 (layer 6 output).

Layers 0-6 receive distillation gradient; 7-9 (P5 conv, C3, SPPF) stay at
init and are trained in Stage 3 (documented in the design note).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "yolov5"))

P4_LAYER = 6  # index of the C3 block at stride 16 in yolov5 backbone yamls


class BackboneStudent(nn.Module):
    def __init__(self, model_yaml="yolov5/models/yolov5s.yaml", nc=10, teacher_dim=1280):
        super().__init__()
        from models.yolo import DetectionModel
        full = DetectionModel(cfg=str(REPO / model_yaml), ch=3, nc=nc)
        self.layers = full.model[: P4_LAYER + 1]  # 0..6 inclusive
        with torch.no_grad():
            p4 = self._forward_backbone(torch.zeros(1, 3, 224, 224))
        self.p4_channels = p4.shape[1]
        self.p4_grid = p4.shape[2]
        self.proj = nn.Conv2d(self.p4_channels, teacher_dim, kernel_size=1)  # discarded after stage 2

    def _forward_backbone(self, x):
        for m in self.layers:  # backbone layers are strictly sequential (f=-1)
            x = m(x)
        return x

    def forward(self, x):
        """returns (B, D_t, g, g) projected student features"""
        return self.proj(self._forward_backbone(x))

    def backbone_state_dict(self):
        """state_dict of layers 0..6 keyed as 'model.<i>.' to match DetectionModel."""
        out = {}
        for i, m in enumerate(self.layers):
            for k, v in m.state_dict().items():
                out[f"model.{i}.{k}"] = v.cpu()
        return out


def cosine_distill_loss(student_feat, teacher_tokens):
    """student_feat: (B, D, gs, gs); teacher_tokens: (B, gt, gt, D).
    Teacher grid is bilinearly resized to the student grid; both L2-normalized;
    loss = mean over positions of (1 - cosine)."""
    B, D, gs, _ = student_feat.shape
    t = teacher_tokens.permute(0, 3, 1, 2).float()  # (B, D, gt, gt)
    if t.shape[-1] != gs:
        t = torch.nn.functional.interpolate(t, size=(gs, gs), mode="bilinear", align_corners=False)
    s = torch.nn.functional.normalize(student_feat.float(), dim=1)
    t = torch.nn.functional.normalize(t, dim=1)
    return (1.0 - (s * t).sum(dim=1)).mean()


def smooth_l1_distill_loss(student_feat, teacher_tokens):
    """fallback loss (design note): smooth-L1 on L2-normalized features."""
    B, D, gs, _ = student_feat.shape
    t = teacher_tokens.permute(0, 3, 1, 2).float()
    if t.shape[-1] != gs:
        t = torch.nn.functional.interpolate(t, size=(gs, gs), mode="bilinear", align_corners=False)
    s = torch.nn.functional.normalize(student_feat.float(), dim=1)
    t = torch.nn.functional.normalize(t, dim=1)
    return torch.nn.functional.smooth_l1_loss(s, t)
