# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-11-26

First public release.

### Added
- **Build** — visual canvas for drawing networks (Input, Dense, Conv2D,
  MaxPool2D, Flatten, Output, and a free-form Custom node), with per-neuron
  weights, bias, and activation editing.
- **Compile** — JSON graph is validated and built into a runnable PyTorch model;
  structural errors are reported against the offending nodes.
- **Train** — dataset upload (CSV / JSON / NPY / NPZ / image ZIP) and built-in
  generators; configurable optimizer, loss, scheduler, epochs, and batch size;
  live training over Server-Sent Events with checkpoints and early stopping.
- **Observe** — run the model on one input and replay the forward pass,
  layer by layer, with per-neuron activations, snapshots, and labelled
  predictions; edit weights live (in place) and re-run.
- **Examples & import/export** — ANN and CNN (incl. AlexNet) example networks and
  a `.tcnet.json` architecture file format.
- **Pretrained** — load real pretrained **AlexNet** or **VGG-16** weights from
  torchvision for ImageNet image classification, with top-k labelled predictions.
- **Developer experience** — one-command runner (`dev.sh`), backend test suite,
  and a backend-served catalog endpoint with a client/server drift guard.

[Unreleased]: https://github.com/Swapnil-Kapale/tensorcanvas/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Swapnil-Kapale/tensorcanvas/releases/tag/v0.1.0
