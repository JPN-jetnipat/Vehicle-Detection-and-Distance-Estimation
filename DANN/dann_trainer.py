"""
Ultralytics trainer subclass that feeds a target-domain batch alongside
each source batch.

Subclassing DetectionTrainer (rather than writing a training loop from
scratch) keeps mosaic, copy_paste, hsv_v, close_mosaic and every other
augmentation byte-identical to the source-only baseline. Any difference in
augmentation between the baseline and the DANN run would be a confound that
cannot be separated from the effect of the adversary.
"""

import torch
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer

from dann_core import DANNDetectionModel, lambda_schedule


def de_parallel(model):
    """
    Unwrap a DataParallel / DistributedDataParallel model.

    Defined locally rather than imported: ultralytics has moved this helper
    between modules across versions, and the implementation is two lines.
    """
    return model.module if hasattr(model, "module") else model


class DANNTrainer(DetectionTrainer):
    """
    Set these before constructing the trainer (class attributes, because
    Ultralytics rejects unknown keys in its args namespace):

        DANNTrainer.target_img_path  path to target-domain images
        DANNTrainer.warmup_epochs    epochs with lambda forced to 0
        DANNTrainer.lambda_max       peak GRL coefficient
        DANNTrainer.focal_gamma      focal loss gamma for the global head
        DANNTrainer.local_weight     weight on the local (shallow) head
        DANNTrainer.global_weight    weight on the global (deep) head
    """

    target_img_path = None
    warmup_epochs = 3
    lambda_max = 1.0
    focal_gamma = 3.0
    local_weight = 1.0
    global_weight = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_loader = None
        self._target_iter = None
        self._step = 0
        self._last_epoch = -1
        self._nan_steps = 0

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DANNDetectionModel(cfg, nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)
        model.focal_gamma = self.focal_gamma
        model.local_weight = self.local_weight
        model.global_weight = self.global_weight
        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        # Report the domain loss in the progress table alongside the
        # detection losses, so a collapse is visible while it happens
        # rather than only at the end of the run. Appending to whatever the
        # parent set avoids hard-coding the detection loss names, which
        # differ between ultralytics versions.
        names = list(getattr(self, "loss_names", ["box_loss", "cls_loss", "dfl_loss"]))
        if "dom_loss" not in names:
            names.append("dom_loss")
        self.loss_names = tuple(names)

    # -- target domain dataloader ------------------------------------------

    def _build_target_loader(self):
        assert self.target_img_path, "DANNTrainer.target_img_path is not set"
        batch_size = self.batch_size
        gs = max(int(de_parallel(self.model).stride.max()), 32)
        dataset = build_yolo_dataset(
            self.args, self.target_img_path, batch_size, self.data,
            mode="train", rect=False, stride=gs,
        )
        # Labels for the target split are loaded by the dataset but never
        # reach the loss -- only batch["img"] is used. This stays an
        # unsupervised setup.
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=True, rank=-1)

    def _next_target_batch(self):
        if self._target_loader is None:
            self._target_loader = self._build_target_loader()
            self._target_iter = iter(self._target_loader)
        try:
            return next(self._target_iter)
        except StopIteration:
            self._target_iter = iter(self._target_loader)
            return next(self._target_iter)

    # -- per-step wiring ----------------------------------------------------

    def _current_lambda(self):
        """Zero during warmup, then ramped over the remaining epochs."""
        if self.epoch < self.warmup_epochs:
            return 0.0
        adv_epochs = max(1, self.args.epochs - self.warmup_epochs)
        steps_per_epoch = max(1, len(self.train_loader))
        done = (self.epoch - self.warmup_epochs) * steps_per_epoch + self._step
        progress = done / (adv_epochs * steps_per_epoch)
        return lambda_schedule(progress, self.lambda_max)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)

        # _step counts steps inside the current epoch; reset on epoch change.
        if self.epoch != self._last_epoch:
            self._last_epoch = self.epoch
            self._step = 0

        model = de_parallel(self.model)
        model.dann_lambda = self._current_lambda()

        target = self._next_target_batch()
        # Identical preprocessing to the source batch. A mismatch here is
        # what produced NaN domain loss in v1.
        batch["target_img"] = target["img"].to(self.device, non_blocking=True).float() / 255.0

        self._step += 1
        return batch