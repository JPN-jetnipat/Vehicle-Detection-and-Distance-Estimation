# Stage 2 Design Note — Feature Distillation (frozen I-JEPA ViT → CSP-Darknet)

*Status: awaiting Kanade's sign-off (brief §7). No Stage-2 code is written until approved.*

## What gets aligned to what

One image, one shared view, two networks:

```
image ──RandomResizedCrop(224, scale 0.3–1.0) + hflip──┬─► frozen ViT teacher ─► tokens (grid G_t × G_t, dim D_t) ─► LayerNorm'd last layer
                                                        └─► YOLOv5s backbone   ─► P4 map (14×14, 256 ch) ─► 1×1 conv proj → D_t
```

- **Teacher output:** final-layer tokens of the **target encoder** (for T1, the released `ijepa_vith14_1k` checkpoint *is* the target encoder). No [cls] token exists — all tokens are patch tokens.
- **Teacher grids:** T1 ViT-H/14 @224 → **16×16, dim 1280**. T2 ViT-B/16 @224 → **14×14, dim 768**. The module takes `(embed_dim, grid)` as parameters — swapping teachers is a config change.
- **Student tap point:** output of backbone layer 6 (the C3 block at stride 16 = **P4**), which at 224-px input is exactly **14×14** — a natural match. For T1, the teacher's 16×16 grid is bilinearly interpolated to 14×14 (interpolate the teacher, not the student: cheaper, and averaging semantic tokens is safer than stretching student features).
- **Why 224 input for the student** (not YOLO's 640): the teacher was pretrained at 224 and its checkpoint has 224 position embeddings; matching keeps teacher features trustworthy and makes the grids line up. The backbone is fully convolutional, so weights transfer to 640-px fine-tuning unchanged.

## Loss

Both feature maps L2-normalized per token, then **cosine distance** `1 − cos(proj(s), t)` averaged over all 196 positions. (Smooth-L1 on normalized features is the documented fallback if cosine stalls; switching is one flag.) Projection is a single 1×1 conv (256 → D_t) that is **discarded after Stage 2** — only backbone weights move to Stage 3.

## What gets trained, what stays random

Gradient reaches backbone layers 0–6. Layers 7–9 (P5 conv, C3, SPPF) receive no distillation signal and stay random-init into Stage 3 (they train there end-to-end). This is the documented "single-level P4" start per the brief; if T1 results disappoint, the first escalation is an auxiliary P5 ↔ pooled-teacher loss, second is P3. Neck + head are always fresh-init in Stage 3, identically across all arms.

## Data & sampling

- Pool: `splits/pretrain.txt` (62,020 unlabeled train images; modelsel excluded, val never).
- Augmentation: RandomResizedCrop(224) + horizontal flip only — identical view to both nets, teacher run on the fly, `torch.no_grad()` + eval mode.
- Sampling: **uniform** (default). Night/rainy oversampling is Stage 1's job (the T2 teacher learns the domain); replicating it in Stage 2 would confound "better teacher" with "different distillation data". Open question flagged below.

## Schedule & budget (A40, bf16 autocast)

- AdamW, lr 1e-3 (cosine to 1e-5, 3-epoch warmup), weight decay 0.05, batch 128 (grad-accum ×2 from 64 if VRAM contended).
- **30 epochs** over 62k images. Throughput estimate: teacher ViT-H forward dominates; expect roughly 250–350 img/s → **~1.5–3 h total for T1**. T2 (ViT-B teacher) is ~5× lighter, well under 1 h. Both far below the 2-day flag threshold. Measured throughput gets logged in the first 10 minutes; deviations >2× get flagged.
- VRAM: ≈ 8–12 GB at batch 128 — fine on a shared 46 GB card, still check `nvidia-smi`.
- Checkpoint every 500 iters + per epoch; auto-resume; TensorBoard + CSV in the run dir; config dump + git hash as in Stage 3.

## How we know it's working / when to stop

1. Distillation loss decreasing (cosine distance from ~1.0 toward ~0.2–0.4 is typical; exact floor doesn't matter, trend does).
2. **Cheap probe every 5 epochs:** linear day-vs-night classifier on pooled student P4 features (labels free from the attribute index). Should climb well above chance and plateau — plateau = safe to stop.
3. Decisive test: Stage 3 fine-tune at 10% labels vs COCO-init at 10%.

## Open items folded into sign-off

a) P4-only start (escalate to P5/P3 only on weak T1 results) — recommended yes.
b) Cosine loss primary, smooth-L1 as fallback flag — recommended yes.
c) Uniform sampling in Stage 2 (no night oversampling here) — recommended yes.
d) 30 epochs / batch 128 / lr 1e-3 defaults — recommended yes.
