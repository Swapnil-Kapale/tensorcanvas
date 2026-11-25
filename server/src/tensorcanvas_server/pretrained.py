"""Load real pretrained weights (torchvision) into a TensorCanvas graph that
matches the reference architecture, in place.

Adding another pretrained model (e.g. ResNet, VGG) is two steps and needs no
changes anywhere else:

  1. Write a ``load_<arch>_into(built, record) -> list[str]`` function here that
     copies the torchvision weights into the graph (see ``load_alexnet_into`` and
     the generic ``copy_conv_linear_in_order`` helper below).
  2. Register it in the ``PRETRAINED`` dict at the bottom of this file.

The API endpoint (``POST /models/{id}/weights/pretrained``) dispatches through
``PRETRAINED`` automatically, so a new entry is immediately usable. You will also
want to add a matching architecture as an example graph in
``client/lib/examples.ts`` so users can draw/import the right shape to load into.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

# A loader takes the built graph + its model record and copies real weights in
# place, returning the list of human-readable class labels for the output units.
PretrainedLoader = Callable[[object, object], list[str]]


def copy_conv_linear_in_order(built, record, tv_model, arch_name: str) -> None:
    """Copy a torchvision model's Conv2d/Linear weights into the built graph's
    conv/linear layers, matched positionally in graph order.

    Shared by every loader: it enforces that the drawn architecture matches the
    reference (same number of conv + linear layers, same shapes) and raises a
    clear ``ValueError`` otherwise. Pooling/activation/flatten layers carry no
    weights and are ignored on both sides.
    """
    tv_convs = [m for m in tv_model.modules() if isinstance(m, nn.Conv2d)]
    tv_lins = [m for m in tv_model.modules() if isinstance(m, nn.Linear)]
    our_convs = [built.core[nid] for nid in built.order if isinstance(built.core.get(nid), nn.Conv2d)]
    our_lins = [built.core[nid] for nid in built.order if isinstance(built.core.get(nid), nn.Linear)]

    if len(our_convs) != len(tv_convs) or len(our_lins) != len(tv_lins):
        raise ValueError(
            f"This network is not {arch_name} — expected "
            f"{len(tv_convs)} conv + {len(tv_lins)} linear layers, found "
            f"{len(our_convs)} + {len(our_lins)}. Load the matching example first."
        )

    with record.train_lock, torch.no_grad():
        for ours, t in [*zip(our_convs, tv_convs), *zip(our_lins, tv_lins)]:
            if tuple(ours.weight.shape) != tuple(t.weight.shape):
                raise ValueError(
                    f"layer shape {tuple(ours.weight.shape)} does not match {arch_name} "
                    f"{tuple(t.weight.shape)} — the architecture must match exactly."
                )
            ours.weight.copy_(t.weight)
            if ours.bias is not None and t.bias is not None:
                ours.bias.copy_(t.bias)


def _set_input_norm(record, weights) -> None:
    """Record the model's expected per-channel input normalization (ImageNet
    mean/std) from its torchvision transform, so image inputs are normalized the
    same way the weights were trained — without it, predictions run
    out-of-distribution and degrade badly."""
    try:
        tf = weights.transforms()
        mean, std = getattr(tf, "mean", None), getattr(tf, "std", None)
        if mean and std:
            record.input_norm = {"mean": [float(m) for m in mean], "std": [float(s) for s in std]}
    except Exception:
        record.input_norm = None


def load_alexnet_into(built, record) -> list[str]:
    """Copy torchvision's pretrained AlexNet weights into the built graph's conv/
    linear layers (in graph order). Returns the ImageNet category names.

    Raises ValueError if the graph is not shaped like torchvision AlexNet.
    """
    from torchvision.models import AlexNet_Weights, alexnet

    weights = AlexNet_Weights.DEFAULT
    tv = alexnet(weights=weights)
    tv.eval()
    copy_conv_linear_in_order(built, record, tv, "torchvision AlexNet")
    _set_input_norm(record, weights)
    return list(weights.meta["categories"])


def load_vgg16_into(built, record) -> list[str]:
    """Copy torchvision's pretrained VGG-16 weights into the built graph's conv/
    linear layers (in graph order). Returns the ImageNet category names.

    Expects the VGG-16 example graph (13 conv + 3 linear, 224×224 input). Raises
    ValueError if the architecture does not match.
    """
    from torchvision.models import VGG16_Weights, vgg16

    weights = VGG16_Weights.DEFAULT
    tv = vgg16(weights=weights)
    tv.eval()
    copy_conv_linear_in_order(built, record, tv, "torchvision VGG-16")
    _set_input_norm(record, weights)
    return list(weights.meta["categories"])


# ── Registry ──────────────────────────────────────────────────────────────────
# Map an architecture name → its loader. Add a new entry to expose a new
# pretrained model through the API; nothing else needs to change.
PRETRAINED: dict[str, PretrainedLoader] = {
    "alexnet": load_alexnet_into,
    "vgg16": load_vgg16_into,
}


def available_pretrained() -> list[str]:
    """Names of the pretrained architectures that can be loaded."""
    return sorted(PRETRAINED)


def load_pretrained_into(arch: str, built, record) -> list[str]:
    """Dispatch to the loader for ``arch``. Raises KeyError if unknown."""
    loader = PRETRAINED[arch.lower()]
    return loader(built, record)
