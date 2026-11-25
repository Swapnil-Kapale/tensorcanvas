"""Load real pretrained weights (torchvision) into a TensorCanvas graph that
matches the reference architecture, in place. Currently: AlexNet (ImageNet)."""

from __future__ import annotations

import torch
from torch import nn


def load_alexnet_into(built, record) -> list[str]:
    """Copy torchvision's pretrained AlexNet weights into the built graph's conv/
    linear layers (in graph order). Returns the ImageNet category names.

    Raises ValueError if the graph is not shaped like torchvision AlexNet.
    """
    from torchvision.models import AlexNet_Weights, alexnet

    weights = AlexNet_Weights.DEFAULT
    tv = alexnet(weights=weights)
    tv.eval()

    tv_convs = [m for m in tv.features if isinstance(m, nn.Conv2d)]
    tv_lins = [m for m in tv.classifier if isinstance(m, nn.Linear)]
    our_convs = [built.core[nid] for nid in built.order if isinstance(built.core.get(nid), nn.Conv2d)]
    our_lins = [built.core[nid] for nid in built.order if isinstance(built.core.get(nid), nn.Linear)]

    if len(our_convs) != len(tv_convs) or len(our_lins) != len(tv_lins):
        raise ValueError(
            "This network is not torchvision AlexNet — expected "
            f"{len(tv_convs)} conv + {len(tv_lins)} linear layers, found "
            f"{len(our_convs)} + {len(our_lins)}. Load the AlexNet example (CNN → AlexNet)."
        )

    with record.train_lock, torch.no_grad():
        for ours, t in [*zip(our_convs, tv_convs), *zip(our_lins, tv_lins)]:
            if tuple(ours.weight.shape) != tuple(t.weight.shape):
                raise ValueError(
                    f"layer shape {tuple(ours.weight.shape)} does not match AlexNet "
                    f"{tuple(t.weight.shape)} — the architecture must match exactly."
                )
            ours.weight.copy_(t.weight)
            if ours.bias is not None and t.bias is not None:
                ours.bias.copy_(t.bias)

    return list(weights.meta["categories"])
