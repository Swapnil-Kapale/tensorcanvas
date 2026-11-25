# Security Policy

TensorCanvas is a **local, single-user development tool**, not a hardened
multi-tenant service. Understanding its threat model is the most important part
of using it safely.

## Threat model & intended use

- The backend is meant to bind to **`127.0.0.1` (localhost) only** (the default).
  It has **no authentication or authorization** — anyone who can reach the port
  can use the full API.
- Building and running a network **executes computation you define** (a PyTorch
  model) in the server process. Training and inference run arbitrary tensor
  graphs on your machine.
- Datasets and model files you upload are **parsed and loaded locally**. Some
  formats (notably NumPy `.npy`/`.npz`) can carry data that should be treated as
  untrusted; only open files you trust.
- Loading pretrained weights **downloads ~233 MB** from the torchvision servers
  on first use.

## Do / don't

- **Do** keep the server on localhost for normal use.
- **Do** treat uploaded datasets/network files as untrusted input — load only
  what you trust.
- **Don't** expose the backend to a public or untrusted network. If you must,
  put it behind your own authenticated reverse proxy and restrict `CORS_ORIGINS`
  — and understand you are accepting the risks above.
- **Don't** run it as a shared/hosted service without adding the access controls
  it deliberately omits.

## Supported versions

This is early-stage software; only the latest release on `main` receives fixes.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue.

- Preferred: open a GitHub [private security advisory](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  on this repository.
- Or email the maintainers: swapnilkapaledev@gmail.com, aryan.tandon03@gmail.com.

We will acknowledge your report and keep you updated on a fix. Thank you for
helping keep users safe.
