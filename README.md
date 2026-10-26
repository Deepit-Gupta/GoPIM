# GOPIM reduced-scale reference implementation

This repository contains the Python graph-training and analytical pipeline
implementation for the reduced-scale GOPIM assignment. It does **not** contain
or execute the upstream DNN+NeuroSim C++ simulator. Reported hardware timings
and energy are analytical estimates, not measured NeuroSim results.

## Install

Requires Python 3.10 or later. From the repository root:

```powershell
python -m pip install -e ".[ogb]"
```

The `ogb` extra installs OGB and PyTorch Geometric, needed to download/load the
graphs and run the training comparisons.

## Reproduce the small-graph comparisons

```powershell
python -m gopim small-graphs --datasets cora,citeseer --epochs 50 --seed 42 --device cpu --output-dir runs/my-run
```

The command downloads the Planetoid graphs if they are not cached, then
compares degree-based ISU, loss-aware ISU, and full vertex updates. It also
writes analytical serial, static equal-replica, and GOPIM allocation
comparisons. Results are CSV/JSON; intermediate model checkpoints are created
under the selected output directory.

The saved single-seed comparison referenced in the report is at
[`runs/loss-aware-isu-evaluation`](./runs/loss-aware-isu-evaluation/).
See [GOPIM.md](./GOPIM.md) for model assumptions and
[PAPER_COMPARISON.md](./PAPER_COMPARISON.md) for the paper trend comparison and
limitations.

## Scope

This is a software/model reference only. The local pipeline model uses
crossbar-pulse latency and first-order array energy estimates; it does not
include peripheral/buffer/interconnect energy, idle energy, full NeuroSim
cycle-level behavior, or an exact SlimGNN implementation. It should not be
presented as a full numerical reproduction of the paper.
