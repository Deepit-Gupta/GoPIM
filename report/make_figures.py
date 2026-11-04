from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATASETS = ("cora", "citeseer")
LABELS = ("Cora", "CiteSeer")
POLICIES = (
    ("full_update", "Full update", "#4C78A8"),
    ("isu", "Degree ISU", "#F58518"),
    ("loss_aware_isu", "Loss-aware ISU", "#54A24B"),
)
SCHEMES = (
    ("serial", "Serial", "#4C78A8"),
    ("slimgnn_like_static_equal_replication", "Static equal-replica", "#F58518"),
    ("gopim_greedy_bottleneck", "GOPIM greedy", "#54A24B"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def make_training_figure() -> None:
    rows = {row["dataset"]: row for row in read_csv(RESULTS / "update_comparison.csv")}
    accuracy = {
        policy: [
            100
            * json.loads(rows[dataset][f"{policy}_final_test"])["acc"]
            for dataset in DATASETS
        ]
        for policy, _, _ in POLICIES
    }
    counts = {
        policy: [
            int(rows[dataset][f"{policy}_total_vertex_updates"])
            / int(rows[dataset]["full_update_total_vertex_updates"])
            for dataset in DATASETS
        ]
        for policy, _, _ in POLICIES
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8), constrained_layout=True)
    x = np.arange(len(DATASETS))
    width = 0.23

    for index, (policy, label, color) in enumerate(POLICIES):
        offset = (index - 1) * width
        bars = axes[0].bar(
            x + offset, accuracy[policy], width, label=label, color=color, edgecolor="black", linewidth=0.35
        )
        axes[0].bar_label(bars, fmt="%.1f", fontsize=6, padding=1)
        bars = axes[1].bar(
            x + offset, counts[policy], width, label=label, color=color, edgecolor="black", linewidth=0.35
        )
        axes[1].bar_label(bars, fmt="%.3f", fontsize=6, padding=1)

    axes[0].set_title("(a) Final test accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[1].set_title("(b) Update work normalized to full update")
    axes[1].set_ylabel("Vertex updates / full-update count")
    axes[1].set_ylim(0, 1.2)
    for ax in axes:
        ax.set_xticks(x, LABELS)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=0.7)
    axes[1].legend(frameon=False, loc="upper right", ncol=1)
    fig.savefig(FIGURES / "training_results.pdf", bbox_inches="tight")
    plt.close(fig)


def make_pipeline_figure() -> None:
    schemes_by_dataset = {}
    energy_by_dataset = {}
    for dataset in DATASETS:
        rows = read_csv(RESULTS / f"{dataset}_pipeline.csv")
        by_scheme = {row["scheme"]: row for row in rows}
        serial_time = float(by_scheme["serial"]["makespan_seconds"])
        serial_energy = float(by_scheme["serial"]["array_dynamic_energy_joules"])
        schemes_by_dataset[dataset] = [
            float(by_scheme[scheme]["makespan_seconds"]) / serial_time
            for scheme, _, _ in SCHEMES
        ]
        energy_by_dataset[dataset] = [
            float(by_scheme[scheme]["array_dynamic_energy_joules"]) / serial_energy
            for scheme, _, _ in SCHEMES
        ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8), constrained_layout=True)
    x = np.arange(len(DATASETS))
    width = 0.23

    for index, (scheme, label, color) in enumerate(SCHEMES):
        offset = (index - 1) * width
        times = [schemes_by_dataset[dataset][index] for dataset in DATASETS]
        energies = [energy_by_dataset[dataset][index] for dataset in DATASETS]
        bars = axes[0].bar(
            x + offset, times, width, label=label, color=color, edgecolor="black", linewidth=0.35
        )
        axes[0].bar_label(bars, fmt="%.2f", fontsize=6, padding=1)
        axes[1].bar(
            x + offset, energies, width, label=label, color=color, edgecolor="black", linewidth=0.35
        )

    axes[0].set_title("(a) Analytical makespan")
    axes[0].set_ylabel("Makespan / serial (lower is better)")
    axes[0].set_ylim(0, 2.7)
    axes[1].set_title("(b) Array energy")
    axes[1].set_ylabel("Energy / serial")
    axes[1].set_ylim(0, 1.25)
    axes[1].text(
        0.5,
        0.12,
        "Equal by model construction",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=7,
        color="#444444",
    )
    for ax in axes:
        ax.set_xticks(x, LABELS)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=0.7)
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=0.7)
    axes[1].legend(frameon=False, loc="upper right")
    fig.savefig(FIGURES / "pipeline_results.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    make_training_figure()
    make_pipeline_figure()
    print(f"Generated figures in {FIGURES}")


if __name__ == "__main__":
    main()
