# GOPIM paper trend comparison

## Source and comparison scope

This comparison uses the GOPIM HPCA 2025 paper by Yang et al.,
[“GOPIM: GCN-Oriented Pipeline Optimization for PIM Accelerators”](https://doi.org/10.1109/HPCA61900.2025.00056)
([author-hosted PDF](https://shuibing9420.github.io/assets/pdf/gopim-hpca25.pdf)).
Paper references below point to its tables, figures, and reported results.

The local experiments are a reduced-scale software/model study, not a
reproduction of the paper's hardware-simulator experiments. The saved training
run uses one seed (42), 50 epochs, and the same Cora/CiteSeer data splits for
degree-based ISU, loss-aware ISU, and full-update controls. Results are in
[`runs/loss-aware-isu-evaluation/update_comparison.json`](./runs/loss-aware-isu-evaluation/update_comparison.json)
and its CSV counterpart.

## What the paper reports

- The paper's overall comparison (Figure 13) reports GOPIM as the fastest and
  most energy-efficient method among its evaluated schemes. Its text reports
  speedups versus Serial of 10.2x–3454.3x (727.6x average) over five OGB
  datasets, and average energy reduction of 4.0x versus Serial. These are
  simulator results under the paper's hardware and workload setup.
- In the paper's technique ablation (Figure 14), adding pipelining, then ISU,
  then GOPIM's learned replica allocation improves speed on DDI; the paper
  reports a 3472.3x speedup for GOPIM versus Serial in that discussion.
- The paper reports that ISU's accuracy effect depends on the dataset (Table V):
  it improves accuracy on DDI, PPA, and proteins, and reduces it slightly on
  COLLAB and arxiv. For Cora, the sparse-graph discussion reports a 0.28%
  accuracy loss and 3460.5x speedup versus Serial, with additional speedups
  versus SlimGNN-like, ReGraphX, and ReFlip.
- The paper uses DDI, COLLAB, PPA, proteins, arxiv, and Cora. CiteSeer is not
  among its evaluated datasets. Its Cora model configuration in Table IV is a
  three-layer, 256-hidden-channel model; the local run uses the repository's
  Planetoid input features, two layers, and 64 hidden channels.

The paper's abstract/conclusion gives a separate headline of up to 191x
speedup and 16.1x energy saving versus state-of-the-art work. The reported
figure-specific speedups are not interchangeable with that headline because
the comparison baselines and plotted experiments differ; this report preserves
the figures' stated scope rather than treating those numbers as one universal
result.

## Local results

| Dataset | Full-update test accuracy | Degree ISU | Loss-aware ISU | Selective update count (each) | Full-update count |
|---|---:|---:|---:|---:|---:|
| Cora | 79.0% | 80.4% (+1.4 pp vs full) | 78.6% (-0.4 pp vs full; -1.8 pp vs degree ISU) | 109,432 | 135,400 |
| CiteSeer | 64.3% | 64.2% (-0.1 pp vs full) | 64.4% (+0.1 pp vs full; +0.2 pp vs degree ISU) | 134,430 | 166,350 |

Both selective policies used 19.2% fewer modeled vertex updates than the
full-update control in these runs. This agrees with the intended update-work
reduction, but the accuracy effect is mixed: loss-aware ISU is not consistently
better than degree ISU or full updates. These are single-seed test results and
do not establish statistical significance.

The local analytical pipeline reports the following makespans:

| Dataset | Serial model | Static equal-replica proxy | GOPIM allocator model |
|---|---:|---:|---:|
| Cora | 3.787 us | 5.563 us (0.681x vs serial) | 6.087 us (0.622x vs serial) |
| CiteSeer | 19.286 us | 38.012 us (0.507x vs serial) | 43.116 us (0.447x vs serial) |

Unlike the paper's reported trend, this analytical model predicts both
pipelined schemes are slower than its Serial model for these two graphs.
Estimated array energy is identical across allocation schemes by construction:
the model holds total workload energy constant and does not model idle power,
replica-copy cost, or allocation-dependent writes. It therefore cannot test
the paper's energy-saving trend. Its modeled totals are about 1.515 mJ for
Cora and 4.784 mJ for CiteSeer, the same for all three allocation schemes
within each graph.

## Why these results differ and are not directly comparable

1. **Different graphs and tasks.** The paper's primary aggregate results use
   OGB link- and node-prediction datasets; the local training comparison uses
   small Cora and CiteSeer node-classification graphs. CiteSeer is not a paper
   benchmark. Cora is shared, but sharing the dataset alone does not make the
   experiments equivalent.
2. **Different training configuration.** The local Cora/CiteSeer runs use two
   layers, hidden width 64, one seed, and 50 epochs. The paper's Cora model
   table specifies three layers and 256 hidden channels; its accuracy figure
   and OGB accuracy table use their own training and evaluation settings.
3. **Different hardware/timing fidelity.** The paper evaluates a detailed
   ReRAM PIM accelerator simulator. The local timing is an analytical
   crossbar-pulse floor, not full NeuroSim cycle/energy output; it excludes
   peripheral circuits, buffer/interconnect timing and energy, leakage, and
   idle power. Consequently, its timing and energy numbers cannot validate the
   paper's measured simulator gains.
4. **Different Serial resource semantics.** The local comparator defines Serial
   as sequential stages that each reuse the full crossbar budget. The paper's
   Serial baseline maps one instance per stage (its DDI allocation table lists
   2,264 crossbars total). The models therefore do not share the same Serial
   resource baseline.
5. **Different pipeline baseline.** The local “SlimGNN-like” row is a static
   equal-replica proxy, not the paper's exact SlimGNN implementation. It also
   does not model the paper's ReGraphX or ReFlip baselines.
6. **Different ISU evaluation.** The local degree-ISU threshold follows the
   paper's sparse-graph 80% policy, but the loss-aware policy is a new
   experiment, not part of the paper. The local update counter is a software
   proxy and does not itself measure ReRAM write energy or elapsed hardware
   time.

The direction of the local pipeline result is thus a finding about the limits
of this analytical model, not evidence that GOPIM is slower than the paper's
simulator or that the paper's result is incorrect. A valid numerical
reproduction would require matching the paper's datasets, model/training
protocol, simulator fidelity, resource accounting, and baselines.

## Bottom line

The local work reproduces the assignment's reduced-scale implementation
requirements and reports an extra loss-aware update policy. It reproduces
neither the paper's absolute metrics nor all of its performance trends. The
supported conclusions are limited to the local single-seed accuracy/update
comparison and the stated analytical pipeline model.
