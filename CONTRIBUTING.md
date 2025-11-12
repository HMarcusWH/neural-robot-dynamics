# Contributing

Thanks for helping improve **Neural Robot Dynamics**! This document covers the
minimum steps we ask every contributor to follow before opening a pull request.

## Test matrix

Run the CPU-accessible unit tests locally before you submit code:

```bash
pytest -q
```

GPU-dependent integration tests are opt-in so CI runners without CUDA do not
attempt to launch Warp. Execute them when you have access to a CUDA-capable
device:

```bash
pytest -q -m gpu
```

The GPU suite contains the ICW smoke tests that exercise the auto backend,
reset behaviour, and telemetry parity. These tests automatically skip if CUDA
is unavailable, but running them manually catches regressions early.

## Formatting and linting

We currently rely on the per-module formatting conventions captured in the
repository. Please match the surrounding style and keep imports sorted.

## Pull request checklist

Before requesting review:

- Ensure telemetry and configuration documentation stays in sync with user
  facing behaviour (e.g., backend aliases and CLI flags).
- Include a brief summary of the changes and the commands you ran in the PR
  description.

Thanks again for contributing!
