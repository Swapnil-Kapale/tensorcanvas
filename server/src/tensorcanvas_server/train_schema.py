"""Pydantic models for the training API (kept separate from the build-only schema)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class EarlyStopping(BaseModel):
    enabled: bool = False
    monitor: Literal["val_loss", "val_acc"] = "val_loss"
    patience: int = 10
    min_delta: float = 0.0
    restore_best: bool = True


class TrainConfig(BaseModel):
    dataset_id: str

    # data
    val_split: float = 0.2
    batch_size: int = 32
    shuffle: bool = True
    standardize: bool = False

    # loss
    loss: Literal["auto", "cross_entropy", "nll", "bce", "bce_logits", "mse", "l1", "smooth_l1"] = "auto"
    label_smoothing: float = 0.0

    # optimizer
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    lr: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 0.0
    nesterov: bool = False
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # lr scheduler
    scheduler: Literal["none", "step", "multistep", "exponential", "cosine", "cosine_warm", "plateau"] = "none"
    step_size: int = 30
    gamma: float = 0.1
    t_max: int = 50
    sched_patience: int = 5
    factor: float = 0.1
    min_lr: float = 0.0
    warmup_epochs: int = 0

    # length / stability
    epochs: int = 100
    grad_accum: int = 1
    grad_clip: Optional[float] = None
    early_stopping: EarlyStopping = Field(default_factory=EarlyStopping)

    # init applied before training
    init: Literal["keep", "xavier", "kaiming", "normal"] = "keep"

    # reproducibility
    seed: int = 42
    deterministic: bool = False

    # logging / streaming
    log_every: int = 10
    frame_every: int = 2
