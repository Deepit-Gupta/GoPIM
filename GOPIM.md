# GOPIM software reference implementation

This workspace contains the Python GOPIM graph-training and analytical
pipeline model. It does not vendor, compile, or execute the upstream
DNN+NeuroSim C++ simulator. The analytical model uses 64x64, 2-bit ReRAM
arrays, 29.31 ns read pulses, 50.88 ns write pulses, and a 128 KiB SRAM global
buffer. GOPIM's virtual architecture models 65,536 tiles, 8 PEs per tile, and
32 crossbars per PE in a 4x8 layout. At 2 bits per cell this represents
exactly 16 GiB of crossbar storage; tiles are virtual resource counts and are
never instantiated.

The graph path implements CO, AG, LC, and GC stages per GCN layer, a two-heap
replica allocator, an execution-time MLP, micro-batch pipeline scheduling, SRAM
weight-update accounting, ReLU, and degree-sorted ISU mapping. Sparse/dense
selection and low-degree update cadence follow the configured 80%/50% and
20-epoch policies. The CLI supports the paper's three link-prediction workloads
and three node-prediction workloads, plus the small Cora and CiteSeer
citation graphs for node-classification experiments.

## Setup and run

Install the base Python dependencies and optional OGB graph loader:

```powershell
python -m pip install -e ".[ogb]"
```

Profile the three requested OGB graph datasets and train the stage-time MLP:

```powershell
python -m gopim profile --datasets ddi,collab,ppa --samples 128
```

Train and evaluate on an OGB dataset (link prediction uses the complete
training-edge split by default):

```powershell
python -m gopim train --dataset ddi --epochs 20 --cpu-threads 4
python -m gopim train --dataset ddi --epochs 20 --cpu-threads 4 --disable-isu
```

Use the second command as the vanilla accuracy control. Keep the seed and all
other options identical. Link training computes one graph embedding pass per
epoch and accumulates the complete edge-loss gradient in bounded chunks
(`--link-edge-batch-size`, default 8192). The explicit `--link-steps-per-epoch`
option is only for smoke tests; it caps positive training edges at 64 per step
and is not suitable for reporting benchmark accuracy. Metrics record the number
of positive/negative training edges and training seconds per epoch.

Export an ISU placement for one graph:

```powershell
python -m gopim isu --dataset ddi --graph-index 0
```

## Small-graph training and pipeline comparison

Run the bounded Cora and CiteSeer experiments (PyG Planetoid datasets):

```powershell
python -m gopim small-graphs --epochs 50 --device cpu
```

This runs paired experiments for each graph with identical seeds, model
settings, and train/validation/test splits: one uses selective updating and
one updates all vertices. The adaptive policy updates the highest-degree 80%
of vertices when average degree is at most 8, otherwise the highest-degree
50%; unselected vertices update every 20 epochs.

The evaluation also includes a new loss-aware ISU policy. It keeps the same
per-epoch update budget but ranks vertices using 75% prior-epoch training-loss
rank and 25% degree rank. Vertices without a training label use degree rank
only; the first epoch therefore starts from the degree ordering. This tests
whether spending the existing update budget on harder training vertices
improves accuracy without increasing the modeled number of updates.
`update_comparison.csv` and `update_comparison.json` compare degree ISU,
loss-aware ISU, and full updates, including final metrics, deltas, and
per-epoch/cumulative update counts. Per-epoch details and checkpoints are saved
under each graph's `isu/`, `loss-aware-isu/`, and `full-update/` folders.

The same command writes an analytical pipeline comparison for each graph.
`serial` executes stages one after another while reusing the full crossbar
budget for each stage; `slimgnn_like_static_equal_replication` is an explicitly
approximate baseline that gives pipeline stages an equal replica count;
`gopim_greedy_bottleneck` uses the GOPIM replica allocator.
These results are analytical estimates, not measured SlimGNN or complete
NeuroSim results. Read energy uses a first-order adaptation of the
conventional-parallel 1T1R crossbar array-capacitance equation; write energy
uses an adaptation of the write-cell and line-energy equations, simplified to
one pulse for a fully updated row. Defaults (including ReRAM resistances,
voltage, pulse widths, 40 nm cell/wire feature size, and 32 nm supply voltage)
are model assumptions based on the referenced NeuroSim configuration; the
upstream source is not included. The report records these parameters and
assumptions.

The reported energy includes the modeled crossbar array line switching and
write-cell dynamic energy. It excludes sense amplifiers, decoders, buffers,
interconnect, leakage, and idle energy. The latency is pulse-time only: 29.31 ns
per modeled read operation and 50.88 ns per write pulse, not full subarray
latency. Active read rows are assumed to be 50%, each modeled write updates
all 64 cells in one row using one pulse, and the mean of on/off resistance is
used for write-cell energy. The `SlimGNN-like` comparison remains a static
equal-replica allocation proxy, not a reproduction of SlimGNN. Use
`pipeline --dataset cora` to run just the analytical comparison; results are
saved as `pipeline_comparison.csv` and `.json`.

These are small-graph experiments only. They do not invoke the NeuroSim C++
simulator and do not claim large-graph or paper-level hardware reproduction.

For a direct comparison with the GOPIM paper's reported accuracy, speedup, and
energy trends—and an explanation of the differences and experiment limits—see
[`PAPER_COMPARISON.md`](./PAPER_COMPARISON.md).

`ddi`, `collab`, and `ppa` use OGB link-prediction tasks (`ogbl-ddi`,
`ogbl-collab`, and `ogbl-ppa`). Link training uses sampled positive/negative
edges from the official training split (complete split by default). Graph
training uses the official train split.
Outputs are written under `runs/` by default.
OGB dataset downloads are managed by OGB/torch-geometric. Validation and test
splits are evaluated with each dataset's official evaluator.
For OGB 1.3.6's protocol-4 cache files, compatibility loading is limited to the
dataset's version-marked OGB cache directory.

## Timing model and scope

The Python profiler labels training examples with a deterministic stage-timing
model driven by the configured ReRAM read/write pulse latencies and crossbar
geometry. The MLP learns from those simulated labels. The analytical pipeline
latency is a crossbar-pulse floor and does not model global-buffer transfer
time. This is a runnable
software reference model, not measured cycle-by-cycle output from the upstream
NeuroSim C++ executable; the upstream V2.0 wrapper is CNN-specific. Matching
paper-reported metrics requires the paper's complete calibration parameters,
dataset preprocessing, training recipe, and a graph-stage adapter to the
NeuroSim core. Results from this implementation should not be presented as
paper-reproduction measurements without that calibration.

The upstream DNN+NeuroSim source is intentionally not included in this
reduced-scale results repository. The hardware equations and parameters used
here are analytical assumptions, not runtime integration with that simulator.
