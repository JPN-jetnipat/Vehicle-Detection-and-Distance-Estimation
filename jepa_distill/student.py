"""YOLOv11 backbone student: build from yolo11s.yaml, tap P4 (layer 6 output).

Port of the YOLOv5 version from the archived project. The tap point survived
the detector change unchanged - verified 2026-09-08 at 224px input:

    layer 4  C3k2  -> (256, 28, 28)   stride 8   (P3)
    layer 5  Conv  -> (256, 14, 14)   stride 16
    layer 6  C3k2  -> (256, 14, 14)   stride 16  (P4)  <-- the tap
    layer 7  Conv  -> (512,  7,  7)   stride 32

256 channels at 14x14 is byte-for-byte the same shape the YOLOv5s student
produced, and the layer index is the same too, so the teacher-side grid
matching in the design note carries over with no change:
  - T1 ViT-H/14 @224 -> 16x16 grid, dim 1280 -> bilinearly resized to 14x14
  - T2 ViT-B/16 @224 -> 14x14 grid, dim  768 -> exact match, no resize

Layers 0-6 receive distillation gradient. Layers 7-10 (P5 Conv, C3k2, SPPF,
C2PSA), neck and head get no distillation signal and are taken from the COCO
checkpoint at graft time - see tools/make_init_weights.py. NOTE for the
write-up: YOLOv11's backbone ends in C2PSA (an attention block YOLOv5 did not
have), which sits *after* our tap and is therefore COCO-initialized in every
arm, identically. It is not part of what JEPA touches.

`student_init`:
  "random" (default, matches the signed-off design) - backbone starts from
      scratch, so the fine-tuned backbone's initialization is purely
      JEPA-derived and the arm is a clean contrast with COCO init.
  "coco" - start the student from yolo11s.pt before distilling. Cheaper to
      converge, but the resulting arm is "COCO then JEPA", not "JEPA", and
      must be labelled that way in the table.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

P4_LAYER = 6  # index of the C3k2 block at stride 16 in the yolo11 backbone yaml


class BackboneStudent(nn.Module):
    def __init__(self, model_yaml="yolo11s.yaml", nc=5, teacher_dim=1280,
                 student_init="random"):
        super().__init__()
        from ultralytics.nn.tasks import DetectionModel

        full = DetectionModel(cfg=model_yaml, ch=3, nc=nc, verbose=False)
        if student_init == "coco":
            from ultralytics.nn.tasks import attempt_load_one_weight
            coco, _ = attempt_load_one_weight("yolo11s.pt")
            transferred = full.load_state_dict(coco.state_dict(), strict=False)
            print(f"student_init=coco: missing={len(transferred.missing_keys)} "
                  f"unexpected={len(transferred.unexpected_keys)}")
        elif student_init != "random":
            raise ValueError(f"student_init must be 'random' or 'coco', got {student_init!r}")

        self.layers = full.model[: P4_LAYER + 1]  # 0..6 inclusive (an nn.Sequential slice)

        # The backbone up to P4 must be strictly sequential for the plain
        # forward below to be correct. YOLOv11's `f` is -1 for all of 0..6,
        # but assert it rather than trust it - a silently wrong forward would
        # produce a plausible-looking loss curve and a garbage backbone.
        for i, m in enumerate(self.layers):
            f = getattr(m, "f", -1)
            assert f == -1, (f"layer {i} ({type(m).__name__}) takes input from {f}, "
                             f"not the previous layer - the sequential forward in "
                             f"BackboneStudent is not valid for this yaml.")

        with torch.no_grad():
            p4 = self._forward_backbone(torch.zeros(1, 3, 224, 224))
        self.p4_channels = p4.shape[1]
        self.p4_grid = p4.shape[2]
        assert self.p4_grid == 14, (f"expected a 14x14 P4 grid at 224px, got {self.p4_grid}"
                                    f" - check that {model_yaml} taps stride 16 at layer {P4_LAYER}")
        # 1x1 projection into the teacher's embedding dim; discarded after stage 2
        self.proj = nn.Conv2d(self.p4_channels, teacher_dim, kernel_size=1)

    def _forward_backbone(self, x):
        for m in self.layers:
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


def _align(student_feat, teacher_tokens):
    """(B,D,gs,gs) + (B,gt,gt,D) -> two L2-normalized (B,D,gs,gs) maps.
    Teacher is resized to the student grid, not the other way round: cheaper,
    and averaging semantic tokens is safer than stretching student features."""
    gs = student_feat.shape[-1]
    t = teacher_tokens.permute(0, 3, 1, 2).float()  # (B, D, gt, gt)
    if t.shape[-1] != gs:
        t = F.interpolate(t, size=(gs, gs), mode="bilinear", align_corners=False)
    return F.normalize(student_feat.float(), dim=1), F.normalize(t, dim=1)


def cosine_distill_loss(student_feat, teacher_tokens):
    """mean over positions of (1 - cosine). Primary loss per the design note."""
    s, t = _align(student_feat, teacher_tokens)
    return (1.0 - (s * t).sum(dim=1)).mean()


def smooth_l1_distill_loss(student_feat, teacher_tokens):
    """fallback loss (design note): smooth-L1 on L2-normalized features."""
    s, t = _align(student_feat, teacher_tokens)
    return F.smooth_l1_loss(s, t)


if __name__ == "__main__":
    # self-check: shapes, tap point, gradient reach, and that the state dict
    # keys line up with what tools/make_init_weights.py expects to graft.
    for dim, grid in [(1280, 16), (768, 14)]:
        st = BackboneStudent(teacher_dim=dim)
        x = torch.zeros(2, 3, 224, 224)
        out = st(x)
        fake_teacher = torch.randn(2, grid, grid, dim)
        loss = cosine_distill_loss(out, fake_teacher)
        loss.backward()
        got_grad = sum(p.grad is not None for p in st.layers.parameters())
        total = sum(1 for _ in st.layers.parameters())
        keys = st.backbone_state_dict()
        print(f"teacher dim={dim} grid={grid}: student P4 {st.p4_channels}ch "
              f"{st.p4_grid}x{st.p4_grid}, proj out {tuple(out.shape)}, "
              f"loss {loss.item():.4f}, grads {got_grad}/{total}, "
              f"{len(keys)} backbone tensors, "
              f"layers {sorted({int(k.split('.')[1]) for k in keys})}")
