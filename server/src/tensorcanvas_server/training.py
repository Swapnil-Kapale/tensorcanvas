"""Training building blocks + the loop. The loop takes a `handle` (RunHandle from
runs.py, duck-typed here) for publishing live events and for pause/stop control,
and the model's ModelRecord for its train_lock (guards weight reads during steps)."""

from __future__ import annotations

import copy
import time

import torch
from torch import nn

WEIGHT_ECHO_LIMIT = 4096


# ── model introspection ──────────────────────────────────────────────────────

def _trailing_act(m: nn.Module) -> str | None:
    if isinstance(m, nn.Sequential) and len(m) > 0:
        m = m[-1]
    if isinstance(m, nn.Softmax):
        return "softmax"
    if isinstance(m, nn.Sigmoid):
        return "sigmoid"
    return None


def terminal_activation(built) -> str | None:
    """The activation applied to the network's final output, if any."""
    for nid in reversed(built.module.order):
        a = _trailing_act(built.module.layers[nid])
        if a:
            return a
    return None


# ── config → torch objects ───────────────────────────────────────────────────

class Objective:
    def __init__(self, fn: nn.Module, kind: str, term: str | None):
        self.fn, self.kind, self.term = fn, kind, term

    def loss(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.kind == "nll":
            return self.fn(torch.log(out.clamp_min(1e-9)), y)
        if self.kind == "cross_entropy":
            return self.fn(out, y)
        if self.kind in ("bce", "bce_logits"):
            return self.fn(out.reshape(out.shape[0], -1)[:, :1], y.float().reshape(-1, 1))
        return self.fn(out, y if y.ndim > 1 else y.reshape(-1, 1).float())


def make_objective(cfg, task: str, out_dim: int, term: str | None) -> tuple[Objective, str]:
    name = cfg.loss
    if name == "auto":
        if task == "regression":
            name = "mse"
        elif out_dim == 1:
            name = "bce" if term == "sigmoid" else "bce_logits"
        else:
            name = "nll" if term == "softmax" else "cross_entropy"
    if name == "cross_entropy":
        return Objective(nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing), "cross_entropy", term), name
    if name == "nll":
        return Objective(nn.NLLLoss(), "nll", term), name
    if name == "bce":
        return Objective(nn.BCELoss(), "bce", term), name
    if name == "bce_logits":
        return Objective(nn.BCEWithLogitsLoss(), "bce_logits", term), name
    if name == "mse":
        return Objective(nn.MSELoss(), "reg", term), name
    if name == "l1":
        return Objective(nn.L1Loss(), "reg", term), name
    if name == "smooth_l1":
        return Objective(nn.SmoothL1Loss(), "reg", term), name
    raise ValueError(f"unknown loss '{name}'")


def make_optimizer(params, cfg):
    o = cfg.optimizer
    if o == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, weight_decay=cfg.weight_decay)
    if o == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, weight_decay=cfg.weight_decay)
    if o == "rmsprop":
        return torch.optim.RMSprop(params, lr=cfg.lr, momentum=cfg.momentum, eps=cfg.eps, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(
        params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov and cfg.momentum > 0,
    )


def make_scheduler(opt, cfg):
    s = cfg.scheduler
    L = torch.optim.lr_scheduler
    if s == "step":
        return L.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma), False
    if s == "multistep":
        return L.MultiStepLR(opt, milestones=[cfg.step_size * (i + 1) for i in range(1, 4)], gamma=cfg.gamma), False
    if s == "exponential":
        return L.ExponentialLR(opt, gamma=cfg.gamma), False
    if s == "cosine":
        return L.CosineAnnealingLR(opt, T_max=max(1, cfg.t_max)), False
    if s == "cosine_warm":
        return L.CosineAnnealingWarmRestarts(opt, T_0=max(1, cfg.t_max)), False
    if s == "plateau":
        return L.ReduceLROnPlateau(opt, patience=cfg.sched_patience, factor=cfg.factor, min_lr=cfg.min_lr), True
    return None, False


def apply_init(built, scheme: str) -> None:
    if scheme == "keep":
        return
    for m in built.module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if scheme == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif scheme == "kaiming":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif scheme == "normal":
                nn.init.normal_(m.weight, std=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ── data helpers ─────────────────────────────────────────────────────────────

def split_data(X, y, val_split, seed, shuffle):
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g) if shuffle else torch.arange(n)
    nval = int(n * val_split)
    vi, ti = idx[:nval], idx[nval:]
    return X[ti], y[ti], X[vi], y[vi]


def accuracy(out, y, task, out_dim, term):
    if task != "classification":
        return None
    if out_dim == 1:
        prob = out.reshape(-1) if term == "sigmoid" else torch.sigmoid(out.reshape(-1))
        return ((prob > 0.5).long() == y.reshape(-1).long()).float().mean().item()
    return (out.argmax(1) == y).float().mean().item()


# ── live-frame helpers ───────────────────────────────────────────────────────

def small_weights(built):
    out = {}
    for nid, core in built.core.items():
        if isinstance(core, nn.Linear) and core.out_features * core.in_features <= WEIGHT_ECHO_LIMIT:
            out[nid] = core.weight.detach().tolist()
    return out


def neuron_mags(built):
    out = {}
    for nid, core in built.core.items():
        if isinstance(core, nn.Linear):
            w = core.weight.detach().abs().mean(1)
            out[nid] = (w / w.max().clamp_min(1e-9)).tolist()
    return out


def grad_norms(built):
    out = {}
    for nid, core in built.core.items():
        if isinstance(core, nn.Linear) and core.weight.grad is not None:
            out[nid] = core.weight.grad.detach().norm().item()
    return out


def decision_boundary(module, Xref, out_dim, term, record, res=48):
    xmin, ymin = Xref.min(0).values.tolist()
    xmax, ymax = Xref.max(0).values.tolist()
    mx, my = (xmax - xmin) * 0.1 + 1e-3, (ymax - ymin) * 0.1 + 1e-3
    xmin, xmax, ymin, ymax = xmin - mx, xmax + mx, ymin - my, ymax + my
    xs = torch.linspace(xmin, xmax, res)
    ys = torch.linspace(ymin, ymax, res)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)
    module.eval()
    with torch.no_grad(), record.train_lock:
        out = module(pts)
    if out_dim == 1:
        prob = out.reshape(-1) if term == "sigmoid" else torch.sigmoid(out.reshape(-1))
        return {"kind": "decision_boundary", "mode": "prob", "res": res,
                "bounds": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
                "values": prob.tolist()}
    return {"kind": "decision_boundary", "mode": "class", "res": res, "n_classes": out_dim,
            "bounds": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
            "values": out.argmax(1).tolist()}


def _evaluate(module, obj, X, y, task, out_dim, term, record):
    if X.shape[0] == 0:
        return 0.0, None
    module.eval()
    with torch.no_grad(), record.train_lock:
        out = module(X)
        loss = float(obj.loss(out, y).item())
        acc = accuracy(out, y, task, out_dim, term)
    return loss, acc


# ── the training loop (runs in a background thread) ──────────────────────────

def run_training(handle, record, built, cfg, X, y, task, n_classes, out_dim) -> None:
    try:
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(1)
        module = built.module
        term = terminal_activation(built)
        apply_init(built, cfg.init)

        Xtr, ytr, Xval, yval = split_data(X, y, cfg.val_split, cfg.seed, cfg.shuffle)
        if cfg.standardize and Xtr.dim() == 2:
            mu = Xtr.mean(0, keepdim=True)
            sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
            Xtr, Xval = (Xtr - mu) / sd, (Xval - mu) / sd

        obj, loss_name = make_objective(cfg, task, out_dim, term)
        opt = make_optimizer(module.parameters(), cfg)
        sched, is_plateau = make_scheduler(opt, cfg)
        two_d = tuple(built.input_shape) == (2,) and task == "classification"

        handle.publish("hello", {
            "run_id": handle.run_id, "status": "running", "loss": loss_name, "task": task,
            "input_shape": list(built.input_shape), "out_dim": out_dim, "epochs": cfg.epochs,
            "n_train": int(Xtr.shape[0]), "n_val": int(Xval.shape[0]), "boundary": two_d,
        })

        g = torch.Generator().manual_seed(cfg.seed)
        n = int(Xtr.shape[0])
        gstep = 0
        best = None
        best_state = None
        bad = 0
        t0 = time.time()

        for epoch in range(cfg.epochs):
            if handle.stopped():
                break
            module.train()
            perm = torch.randperm(n, generator=g) if cfg.shuffle else torch.arange(n)
            opt.zero_grad()
            last_loss = 0.0
            for bi in range(0, n, cfg.batch_size):
                handle.wait_if_paused()
                if handle.stopped():
                    break
                idx = perm[bi : bi + cfg.batch_size]
                with record.train_lock:
                    out = module(Xtr[idx])
                    loss = obj.loss(out, ytr[idx]) / cfg.grad_accum
                    loss.backward()
                    if (gstep + 1) % max(1, cfg.grad_accum) == 0:
                        if cfg.grad_clip:
                            nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
                        opt.step()
                        opt.zero_grad()
                last_loss = float(loss.item() * cfg.grad_accum)
                gstep += 1
                if gstep % max(1, cfg.log_every) == 0:
                    sps = (gstep * cfg.batch_size) / max(1e-6, time.time() - t0)
                    handle.publish("metrics", {
                        "epoch": epoch, "global_step": gstep, "loss": last_loss,
                        "lr": opt.param_groups[0]["lr"], "samples_per_sec": sps,
                    }, droppable=True)
            if handle.stopped():
                break

            tr_loss, tr_acc = _evaluate(module, obj, Xtr, ytr, task, out_dim, term, record)
            va_loss, va_acc = _evaluate(module, obj, Xval, yval, task, out_dim, term, record)
            if Xval.shape[0] == 0:
                va_loss, va_acc = tr_loss, tr_acc
            lr = opt.param_groups[0]["lr"]
            if sched is not None:
                sched.step(va_loss) if is_plateau else sched.step()

            ckpt_id = handle.checkpoint(module, epoch, va_loss, va_acc)
            handle.publish("epoch", {
                "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": va_loss, "val_acc": va_acc, "lr": lr, "ckpt_id": ckpt_id,
                "grad_norms": grad_norms(built), "weights": small_weights(built),
                "neuron_mags": neuron_mags(built),
            })
            if two_d and cfg.frame_every and epoch % cfg.frame_every == 0:
                db = decision_boundary(module, torch.cat([Xtr, Xval]) if Xval.shape[0] else Xtr, out_dim, term, record)
                handle.publish("frame", {"epoch": epoch, **db}, droppable=True)

            es = cfg.early_stopping
            if es.enabled:
                monitor = va_loss if es.monitor == "val_loss" else -((va_acc or 0.0))
                if best is None or monitor < best - es.min_delta:
                    best, bad = monitor, 0
                    best_state = copy.deepcopy(module.state_dict())
                else:
                    bad += 1
                    if bad >= es.patience:
                        handle.note_status("early_stopped")
                        break

        if cfg.early_stopping.enabled and cfg.early_stopping.restore_best and best_state is not None:
            with record.train_lock:
                module.load_state_dict(best_state)
        handle.finish()
    except Exception as e:  # any training error becomes a terminal error event
        handle.error(f"{type(e).__name__}: {e}")
