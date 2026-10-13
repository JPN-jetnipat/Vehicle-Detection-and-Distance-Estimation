"""
DANN for YOLO11 detection — core components.

Design references:
  [1] Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation",
      ICML 2015. Gradient Reversal Layer and the lambda ramp-up schedule.
  [2] Chen et al., "Domain Adaptive Faster R-CNN for Object Detection in the
      Wild", CVPR 2018. Adversarial alignment applied to a detector.
  [3] Saito et al., "Strong-Weak Distribution Alignment for Adaptive Object
      Detection", CVPR 2019. Strong local alignment on shallow features +
      weak global alignment on deep features. This is the layout used here.
  [4] Hsu et al., "Every Pixel Matters", ECCV 2020. Confirms the approach
      transfers to anchor-free detectors.

Why two heads instead of one:
  A day->night shift is mostly low-level (illumination, sensor noise, colour
  cast). SWDA showed that forcing deep semantic features to align strongly
  degrades detection, while aligning shallow features strongly is what
  actually helps. A single deep head -- the v1/v2 layout -- is the
  configuration SWDA reports as weakest.

YOLO11s layer map (verified on this checkpoint):
    0-10  backbone   (every layer has f == -1, so it runs sequentially)
   11-22  neck
      23  Detect
Local head attaches at layer 4 (stride 8, shallow).
Global head attaches at layer 10 (C2PSA, deepest backbone layer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.tasks import DetectionModel

# Backbone layer indices used for domain alignment.
LOCAL_LAYER = 4      # shallow: texture / illumination statistics
GLOBAL_LAYER = 10    # deep: semantic content
BACKBONE_END = 10    # last backbone layer index (inclusive)


# ---------------------------------------------------------------------------
# Gradient Reversal Layer  [1]
# ---------------------------------------------------------------------------

class _GradientReversal(torch.autograd.Function):
    """Identity forward, negated-and-scaled gradient backward."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        # view_as keeps this in the graph; a plain return would still work but
        # this makes the node explicit and easier to spot in graph dumps.
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd):
    return _GradientReversal.apply(x, lambd)


def lambda_schedule(progress, lambda_max=1.0):
    """
    Ramp lambda from 0 to lambda_max over training  [1].

    progress: float in [0, 1], fraction of adversarial training completed.

    Opening the adversary at full strength from step 0 destroys the features
    before the domain heads have learned anything useful, which is a common
    way for DANN runs to collapse.
    """
    p = float(max(0.0, min(1.0, progress)))
    return lambda_max * (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))).item() - 1.0)


# ---------------------------------------------------------------------------
# Domain classifiers
# ---------------------------------------------------------------------------

class LocalDomainHead(nn.Module):
    """
    Pixel-wise domain classifier on shallow features (strong alignment)  [3].

    Uses 1x1 convolutions only, so each spatial position is classified from
    its own local receptive field. Trained with a least-squares loss, which
    SWDA found more stable than cross-entropy for the local head.
    """

    def __init__(self, in_channels, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)  # logits, shape (B, 1, H, W)

    @staticmethod
    def loss(logits, domain_label):
        """Least-squares loss. domain_label: 0.0 for source, 1.0 for target."""
        prob = torch.sigmoid(logits)
        return ((prob - domain_label) ** 2).mean()


class GlobalDomainHead(nn.Module):
    """
    Image-level domain classifier on deep features (weak alignment)  [3].

    Focal loss down-weights examples the discriminator already separates
    easily, so the backbone is not pushed to erase the deep semantic
    differences it needs for detection.
    """

    def __init__(self, in_channels, hidden=256, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.net(pooled)  # logits, shape (B, 1)

    @staticmethod
    def loss(logits, domain_label, gamma=3.0):
        """Focal binary cross-entropy. domain_label: 0.0 source, 1.0 target."""
        prob = torch.sigmoid(logits)
        p_t = prob * domain_label + (1.0 - prob) * (1.0 - domain_label)
        p_t = p_t.clamp(min=1e-7, max=1.0 - 1e-7)
        return (-((1.0 - p_t) ** gamma) * torch.log(p_t)).mean()


# ---------------------------------------------------------------------------
# Detection model with domain heads attached
# ---------------------------------------------------------------------------

class DANNDetectionModel(DetectionModel):
    """
    YOLO11 detection model with two domain classifiers hanging off the
    backbone through a Gradient Reversal Layer.

    The domain heads are registered as submodules so they are picked up by
    the standard Ultralytics optimizer builder. Ganin [1] and SWDA [3] both
    train the discriminators with the same optimizer as the feature
    extractor; splitting them adds failure modes without adding benefit.
    """

    # Class-level defaults. DetectionModel.__init__ runs a forward pass to
    # compute strides, which reaches _predict_once before our __init__ body
    # has executed, so these must exist on the class, not just the instance.
    _capture = False
    _captured = {}
    dann_lambda = 0.0
    focal_gamma = 3.0
    local_weight = 1.0
    global_weight = 1.0

    def __init__(self, cfg="yolo11s.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        # Confirm the backbone really is a straight chain, so running
        # self.model[:BACKBONE_END + 1] sequentially is valid.
        for i in range(BACKBONE_END + 1):
            f = self.model[i].f
            assert f == -1, (
                f"backbone layer {i} takes input from {f}, not the previous "
                f"layer; forward_backbone() would be wrong for this model"
            )

        c_local, c_global = self._probe_channels(ch)
        self.domain_local = LocalDomainHead(c_local)
        self.domain_global = GlobalDomainHead(c_global)

    # -- structure helpers --------------------------------------------------

    def _probe_channels(self, ch):
        """Run a dummy tensor through the backbone to read real channel counts."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, ch, 256, 256)
            f_local, f_global = self.forward_backbone(dummy)
        if was_training:
            self.train()
        return f_local.shape[1], f_global.shape[1]

    def forward_backbone(self, x):
        """
        Run layers 0..BACKBONE_END only and return (local_feat, global_feat).

        Used for the target-domain pass, which needs no neck and no head.
        """
        f_local = None
        for i in range(BACKBONE_END + 1):
            x = self.model[i](x)
            if i == LOCAL_LAYER:
                f_local = x
        return f_local, x

    def _predict_once(self, x, profile=False, embed=None, **kwargs):
        """
        Same as the Ultralytics implementation, plus feature capture.

        The signature mirrors ultralytics 8.4.x, which calls this
        positionally as _predict_once(x, profile, embed). **kwargs absorbs
        any extra keyword (e.g. visualize) other versions may pass.

        Falls back to the parent implementation whenever profiling or
        embedding extraction is requested, so those paths stay exactly as
        upstream defines them.
        """
        if profile or embed:
            return super()._predict_once(x, profile, embed, **kwargs)

        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            if self._capture and m.i in (LOCAL_LAYER, GLOBAL_LAYER):
                self._captured[m.i] = x
        return x

    # -- loss ---------------------------------------------------------------

    def domain_loss(self, f_local, f_global, domain_label):
        """Adversarial loss for one domain's features."""
        lam = self.dann_lambda

        local_logits = self.domain_local(grad_reverse(f_local, lam))
        global_logits = self.domain_global(grad_reverse(f_global, lam))

        l_local = LocalDomainHead.loss(local_logits, domain_label)
        l_global = GlobalDomainHead.loss(global_logits, domain_label, self.focal_gamma)

        return self.local_weight * l_local + self.global_weight * l_global

    @staticmethod
    def _with_domain_item(det_items, value):
        """
        Append the domain loss to whatever container Ultralytics uses for
        loss_items. Older versions return a tensor, 8.4.x returns a dict, so
        both forms are handled rather than assuming one.
        """
        if isinstance(det_items, dict):
            out = dict(det_items)
            out["dom_loss"] = value
            return out
        return torch.cat([det_items, value.view(1).to(det_items.device)])

    def loss(self, batch, preds=None):
        """
        Detection loss on the source batch plus domain loss on both domains.

        batch["target_img"] is injected by DANNTrainer.preprocess_batch and is
        the only thing taken from the target domain. Target labels are never
        read here -- this stays an unsupervised adaptation setup.
        """
        # --- source: detection loss, capturing backbone features on the way
        self._captured = {}
        self._capture = True
        det_loss, det_items = super().loss(batch, preds)
        self._capture = False

        target_img = batch.get("target_img", None)
        if target_img is None or LOCAL_LAYER not in self._captured:
            # No target batch (e.g. validation) -- report zero domain loss.
            zero = torch.zeros((), device=det_loss.device, dtype=det_loss.dtype)
            return det_loss, self._with_domain_item(det_items, zero)

        src_local = self._captured[LOCAL_LAYER]
        src_global = self._captured[GLOBAL_LAYER]

        # --- target: backbone only, no neck, no head, no labels
        tgt_local, tgt_global = self.forward_backbone(target_img)

        d_src = self.domain_loss(src_local, src_global, domain_label=0.0)
        d_tgt = self.domain_loss(tgt_local, tgt_global, domain_label=1.0)
        d_loss = 0.5 * (d_src + d_tgt)

        # NaN guard: v1 collapsed here. Drop the domain term for this step
        # rather than poisoning every weight in the network.
        if not torch.isfinite(d_loss):
            d_loss = torch.zeros((), device=det_loss.device, dtype=det_loss.dtype)

        batch_size = batch["img"].shape[0]
        total = det_loss + d_loss * batch_size

        return total, self._with_domain_item(det_items, d_loss.detach())