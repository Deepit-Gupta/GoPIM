import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .allocator import allocate_replicas
from .benchmark import compare_allocation_schemes
from .config import HardwareConfig
from .dataset import PAPER_WORKLOADS, load_ogb_dataset
from .hardware import GOPIMDataflow, StageTimingModel
from .isu import build_isu_plan, loss_aware_update_mask
from .model import GOPIMGCN, GOPIMLinkGCN
from .model import GOPIMNodeGCN
from .pipeline import PipelineScheduler
from .predictor import (
    ProfileRecord,
    StageTimePredictor,
    load_profile_csv,
    profile_with_timing_model,
)
from .stages import four_stages_per_layer


def _edge_array(graph) -> np.ndarray:
    return graph.edge_index.detach().cpu().numpy()


def _stages_for_graph(graph, layer_dimensions):
    return four_stages_per_layer(
        layer_dimensions,
        int(graph.num_nodes),
        int(graph.edge_index.size(1)),
    )


def _profile_records(
    graph_dataset,
    dataset_name: str,
    layer_dimensions,
    sample_limit: int,
    config: HardwareConfig,
) -> list[ProfileRecord]:
    if graph_dataset.is_link_prediction or graph_dataset.is_node_prediction:
        return profile_with_timing_model(
            _stages_for_graph(graph_dataset.graph, layer_dimensions),
            dataset_name,
            config,
            config.micro_batch_size,
        )
    records: list[ProfileRecord] = []
    train_indices = graph_dataset.splits["train"]
    count = min(sample_limit, len(train_indices))
    for index in train_indices[:count]:
        graph = graph_dataset.dataset[int(index)]
        records.extend(
            profile_with_timing_model(
                _stages_for_graph(graph, layer_dimensions),
                dataset_name,
                config,
                config.micro_batch_size,
            )
        )
    return records


def _save_profiles(path: Path, records: list[ProfileRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "dataset",
                "stage_id",
                "duration_seconds",
                "R_IFM_CO",
                "C_IFM_CO",
                "R_W_CO",
                "C_W_CO",
                "R_A_AG",
                "C_A_AG",
                "R_F_AG",
                "C_F_AG",
                "sparsity",
                "layer",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.dataset,
                    record.stage_id,
                    record.duration_seconds,
                    *record.features,
                ]
            )


def _fit_or_load_predictor(
    graph_dataset,
    layer_dimensions,
    config: HardwareConfig,
    output_dir: Path,
    profile_samples: int,
    predictor_path: Path | None = None,
) -> StageTimePredictor:
    model_path = predictor_path or output_dir / "execution_time_mlp.pt"
    if model_path.exists():
        return StageTimePredictor.load(model_path)
    records = _profile_records(
        graph_dataset,
        graph_dataset.name,
        layer_dimensions,
        profile_samples,
        config,
    )
    predictor = StageTimePredictor()
    predictor.fit(records)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(model_path)
    _save_profiles(output_dir / "stage_profiles.csv", records)
    return predictor


def _target_tensor(labels: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    labels = labels.to(output.device)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    return labels.to(output.dtype)


def _loss(output: torch.Tensor, labels: torch.Tensor, task_type: str) -> torch.Tensor:
    target = _target_tensor(labels, output)
    valid = torch.isfinite(target)
    if not valid.any():
        raise ValueError("A training batch contains no valid graph labels")
    if "multiclass" in task_type:
        class_labels = labels.to(output.device).reshape(-1).long()
        valid_labels = class_labels >= 0
        if not valid_labels.any():
            raise ValueError("A training batch contains no valid class labels")
        return nn.functional.cross_entropy(output[valid_labels], class_labels[valid_labels])
    if "classification" in task_type:
        return nn.functional.binary_cross_entropy_with_logits(
            output[valid], target[valid]
        )
    return nn.functional.mse_loss(output[valid], target[valid])


def _evaluate(graph_dataset, model, split: str, batch_size: int, device):
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    indices = graph_dataset.splits[split].tolist()
    loader = DataLoader(
        Subset(graph_dataset.dataset, indices),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            labels.append(batch.y.detach().cpu().numpy())
            if "multiclass" in graph_dataset.task_type:
                output = output.argmax(dim=-1, keepdim=True)
            elif "classification" in graph_dataset.task_type:
                output = output.sigmoid()
            predictions.append(output.detach().cpu().numpy())
    evaluation = graph_dataset.evaluator.eval(
        {
            "y_true": np.concatenate(labels, axis=0),
            "y_pred": np.concatenate(predictions, axis=0),
        }
    )
    return {key: float(value) for key, value in evaluation.items()}


def _write_isu_mapping(path: Path, graph, config: HardwareConfig):
    plan = build_isu_plan(
        _edge_array(graph),
        int(graph.num_nodes),
        config.total_crossbars,
        sparse_threshold=config.sparse_average_degree_threshold,
        sparse_fraction=config.sparse_update_fraction,
        dense_fraction=config.dense_update_fraction,
        low_degree_update_period=config.low_degree_update_period,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["crossbar_id", "vertex_id", "degree", "selected", "update_period"]
        )
        vertex_crossbar = {
            vertex_id: crossbar_id
            for crossbar_id, vertices in plan.assignments.items()
            for vertex_id in vertices
        }
        for vertex_id in plan.ordered_vertices:
            selected = vertex_id in vertex_crossbar
            writer.writerow(
                [
                    vertex_crossbar.get(vertex_id, ""),
                    vertex_id,
                    plan.degrees[vertex_id],
                    selected,
                    1 if selected else plan.low_degree_update_period,
                ]
            )
    return plan


def profile_command(args) -> None:
    config = HardwareConfig()
    if args.measured_profiles:
        all_records = load_profile_csv(args.measured_profiles)
    else:
        all_records: list[ProfileRecord] = []
        for dataset_name in args.datasets.split(","):
            graph_dataset = load_ogb_dataset(dataset_name.strip(), args.data_root)
            workload = PAPER_WORKLOADS.get(graph_dataset.name)
            input_width = max(
                1,
                graph_dataset.num_node_features
                if graph_dataset.is_node_prediction
                else (workload["input"] if workload else graph_dataset.num_node_features),
            )
            hidden_width = args.hidden or (workload["hidden"] if workload else 64)
            layer_count = workload["layers"] if workload else 2
            if graph_dataset.is_node_prediction:
                output_width = (
                    graph_dataset.num_classes
                    if graph_dataset.num_classes
                    else (workload["output"] if workload else hidden_width)
                )
                widths = (
                    [input_width]
                    + [hidden_width] * (layer_count - 1)
                    + [output_width]
                )
            else:
                widths = [input_width] + [hidden_width] * layer_count
            layer_dimensions = list(zip(widths[:-1], widths[1:]))
            all_records.extend(
                _profile_records(
                    graph_dataset,
                    graph_dataset.name,
                    layer_dimensions,
                    args.samples,
                    config,
                )
            )
    output_dir = Path(args.output_dir)
    if not all_records:
        raise ValueError("No workload profiles were generated")
    _save_profiles(output_dir / "stage_profiles.csv", all_records)
    predictor = StageTimePredictor()
    predictor.fit(all_records, epochs=args.mlp_epochs)
    predictor.save(output_dir / "execution_time_mlp.pt")
    print(
        json.dumps(
            {
                "datasets": sorted({record.dataset for record in all_records}),
                "profile_records": len(all_records),
                "output": str(output_dir.resolve()),
            },
            indent=2,
        )
    )


def _train_epoch(
    args,
    graph_dataset,
    model,
    optimizer,
    predictor,
    allocation,
    scheduler,
    dataflow,
    config,
    epoch,
    device,
):
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    indices = graph_dataset.splits["train"].tolist()
    loader = DataLoader(
        Subset(graph_dataset.dataset, indices),
        batch_size=config.micro_batch_size,
        shuffle=True,
    )
    model.train()
    loss_total = 0.0
    graph_total = 0
    for batch_index, batch in enumerate(loader):
        batch = batch.to(device)
        update_mask = (
            None
            if getattr(args, "disable_isu", False)
            else _batch_update_mask(batch, epoch, config)
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, dataflow=dataflow, gradient_mask=update_mask)
        loss = _loss(output, batch.y, graph_dataset.task_type)
        loss.backward()
        optimizer.step()
        updated_count = (
            int(update_mask.sum().item())
            if update_mask is not None
            else int(batch.num_nodes)
        )
        dataflow.record_backward(
            model.layer_dimensions,
            updated_count,
            int(batch.x.size(1)) if getattr(batch, "x", None) is not None else 1,
        )

        stages = _stages_for_graph(batch, model.layer_dimensions)
        predicted = predictor.predict_stages(stages)
        durations = [
            (
                stage.stage_id,
                predicted[stage.stage_id]
                / allocation.allocations[stage.stage_id].replicas,
            )
            for stage in stages
        ]
        scheduler.schedule(epoch * len(loader) + batch_index, durations)

        graph_count = int(getattr(batch, "num_graphs", 1))
        loss_total += float(loss.item()) * graph_count
        graph_total += graph_count
    return loss_total / max(1, graph_total)


def _vertex_update_mask(graph, epoch: int, config: HardwareConfig) -> torch.Tensor:
    plan = build_isu_plan(
        _edge_array(graph),
        int(graph.num_nodes),
        config.total_crossbars,
        sparse_threshold=config.sparse_average_degree_threshold,
        sparse_fraction=config.sparse_update_fraction,
        dense_fraction=config.dense_update_fraction,
        low_degree_update_period=config.low_degree_update_period,
    )
    return torch.tensor(
        [plan.should_update(vertex, epoch) for vertex in range(int(graph.num_nodes))],
        dtype=torch.bool,
        device=graph.edge_index.device,
    )


def _batch_update_mask(batch, epoch: int, config: HardwareConfig) -> torch.Tensor:
    if getattr(batch, "batch", None) is None:
        return _vertex_update_mask(batch, epoch, config)
    return torch.cat(
        [
            _vertex_update_mask(graph, epoch, config)
            for graph in batch.to_data_list()
        ]
    )


def _link_edges(split: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    if "edge" in split:
        positive = torch.as_tensor(split["edge"], dtype=torch.long, device=device)
    elif "source_node" in split and "target_node" in split:
        source = torch.as_tensor(split["source_node"], dtype=torch.long, device=device)
        target = torch.as_tensor(split["target_node"], dtype=torch.long, device=device)
        positive = torch.stack((source.reshape(-1), target.reshape(-1)), dim=-1)
    else:
        raise ValueError("OGB link split has no supported positive-edge fields")
    if "edge_neg" in split:
        negative = torch.as_tensor(split["edge_neg"], dtype=torch.long, device=device)
        if negative.ndim < 2 or negative.size(-1) != 2:
            raise ValueError("OGB negative edges must have a final dimension of 2")
    elif "target_node_neg" in split and "source_node" in split:
        source = torch.as_tensor(split["source_node"], dtype=torch.long, device=device)
        target = torch.as_tensor(
            split["target_node_neg"], dtype=torch.long, device=device
        )
        source = source.reshape(-1, *([1] * (target.ndim - 1))).expand_as(target)
        negative = torch.stack((source, target), dim=-1)
    else:
        raise ValueError("OGB link split has no supported negative-edge fields")
    return positive.reshape(-1, 2), negative


def _evaluate_link_dataset(graph_dataset, model, graph, split_name: str, device):
    positive, negative = _link_edges(graph_dataset.splits[split_name], device)
    model.eval()
    with torch.no_grad():
        embeddings = model(graph)
        positive_scores = (
            embeddings[positive[:, 0]] * embeddings[positive[:, 1]]
        ).sum(dim=-1)
        negative_scores = (
            embeddings[negative[..., 0]] * embeddings[negative[..., 1]]
        ).sum(dim=-1)
    return {
        key: float(value)
        for key, value in graph_dataset.evaluator.eval(
            {
                "y_pred_pos": positive_scores.cpu().numpy(),
                "y_pred_neg": negative_scores.cpu().numpy(),
            }
        ).items()
    }


def _train_link_dataset(
    args,
    graph_dataset,
    predictor,
    allocation,
    output_dir: Path,
    config: HardwareConfig,
    device,
) -> None:
    from torch_geometric.data import Data
    from torch_geometric.utils import negative_sampling, to_undirected

    graph = graph_dataset.graph
    train_edges = torch.as_tensor(
        graph_dataset.splits["train"]["edge"], dtype=torch.long
    ).t().contiguous()
    graph = Data(
        x=graph.x,
        edge_index=to_undirected(train_edges),
        num_nodes=int(graph.num_nodes),
    ).to(device)
    model = GOPIMLinkGCN(
        args.model_config["input"],
        args.model_config["hidden"],
        layer_count=args.model_config["layers"],
        dropout=args.model_config["dropout"],
        num_nodes=int(graph.num_nodes) if graph.x is None else None,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.model_config["learning_rate"]
    )
    training_edges = graph_dataset.splits["train"]["edge"]
    training_edges = torch.as_tensor(training_edges, dtype=torch.long, device=device)
    isu_plan = build_isu_plan(
        graph.edge_index.detach().cpu().numpy(),
        int(graph.num_nodes),
        config.total_crossbars,
        sparse_threshold=config.sparse_average_degree_threshold,
        sparse_fraction=config.sparse_update_fraction,
        dense_fraction=config.dense_update_fraction,
        low_degree_update_period=config.low_degree_update_period,
    )
    scheduler = PipelineScheduler()
    dataflow = GOPIMDataflow(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_batch_size = getattr(args, "link_edge_batch_size", 8192)
    if edge_batch_size <= 0:
        raise ValueError("link_edge_batch_size must be positive")
    positive_edge_count = training_edges.size(0)
    link_step_limit = getattr(args, "link_steps_per_epoch", None)
    if link_step_limit is not None:
        positive_edge_count = min(
            positive_edge_count,
            link_step_limit * config.micro_batch_size,
        )
    run_config = {
        "dataset": graph_dataset.name,
        "seed": args.seed,
        "epochs": args.epochs,
        "positive_train_edges_available": int(training_edges.size(0)),
        "positive_train_edges_per_epoch": int(positive_edge_count),
        "negative_train_edges_per_epoch": int(positive_edge_count),
        "edge_batch_size": int(edge_batch_size),
        "isu_enabled": not getattr(args, "disable_isu", False),
        "model": args.model_config,
        "device": str(device),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    with (output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "validation",
                "test",
                "positive_train_edges",
                "negative_train_edges",
                "train_seconds",
            ]
        )
        for epoch in range(args.epochs):
            train_started = time.perf_counter()
            stage_specs = _stages_for_graph(graph, model.layer_dimensions)
            stage_predictions = predictor.predict_stages(stage_specs)
            stage_durations = [
                (
                    stage.stage_id,
                    stage_predictions[stage.stage_id]
                    / allocation.allocations[stage.stage_id].replicas,
                )
                for stage in stage_specs
            ]
            if getattr(args, "disable_isu", False):
                gradient_mask = None
                updated_vertex_count = int(graph.num_nodes)
            else:
                gradient_mask = torch.tensor(
                    [
                        isu_plan.should_update(vertex_id, epoch)
                        for vertex_id in range(int(graph.num_nodes))
                    ],
                    dtype=torch.bool,
                    device=device,
                )
                updated_vertex_count = int(gradient_mask.sum().item())

            model.train()
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(graph, dataflow=dataflow, gradient_mask=gradient_mask)
            edge_embeddings = embeddings.detach().requires_grad_(True)
            edge_order = torch.randperm(training_edges.size(0), device=device)
            positive_edges = training_edges.index_select(
                0, edge_order[:positive_edge_count]
            )
            negative_edges = negative_sampling(
                graph.edge_index,
                num_nodes=int(graph.num_nodes),
                num_neg_samples=positive_edge_count,
            ).t().contiguous()
            if negative_edges.size(0) != positive_edge_count:
                raise RuntimeError(
                    "Negative sampler returned fewer edges than requested: "
                    f"{negative_edges.size(0)} < {positive_edge_count}"
                )
            total_examples = positive_edge_count + negative_edges.size(0)
            if total_examples == 0:
                raise RuntimeError("The training split contains no link examples")
            epoch_loss = 0.0
            for start in range(0, positive_edge_count, edge_batch_size):
                end = min(start + edge_batch_size, positive_edge_count)
                positive = positive_edges[start:end]
                negative = negative_edges[start:end]
                positive_scores = (
                    edge_embeddings[positive[:, 0]]
                    * edge_embeddings[positive[:, 1]]
                ).sum(dim=-1)
                negative_scores = (
                    edge_embeddings[negative[:, 0]]
                    * edge_embeddings[negative[:, 1]]
                ).sum(dim=-1)
                loss_sum = nn.functional.binary_cross_entropy_with_logits(
                    positive_scores,
                    torch.ones_like(positive_scores),
                    reduction="sum",
                ) + nn.functional.binary_cross_entropy_with_logits(
                    negative_scores,
                    torch.zeros_like(negative_scores),
                    reduction="sum",
                )
                loss = loss_sum / total_examples
                loss.backward()
                epoch_loss += float(loss.detach().item())

            embeddings.backward(edge_embeddings.grad)
            optimizer.step()
            dataflow.record_backward(
                model.layer_dimensions,
                updated_vertex_count,
                args.model_config["input"],
            )
            scheduler.schedule(epoch, stage_durations)
            train_seconds = time.perf_counter() - train_started

            if (epoch + 1) % args.eval_every == 0:
                validation = _evaluate_link_dataset(
                    graph_dataset, model, graph, "valid", device
                )
                test = _evaluate_link_dataset(
                    graph_dataset, model, graph, "test", device
                )
            else:
                validation, test = {}, {}
            writer.writerow(
                [
                    epoch + 1,
                    epoch_loss,
                    json.dumps(validation, sort_keys=True),
                    json.dumps(test, sort_keys=True),
                    positive_edge_count,
                    negative_edges.size(0),
                    train_seconds,
                ]
            )
            file.flush()
            print(
                f"epoch={epoch + 1} loss={epoch_loss:.6f} "
                f"positive_edges={positive_edge_count} "
                f"train_seconds={train_seconds:.2f} "
                f"validation={validation} test={test}"
            )

    allocation_summary = {
        "crossbars_used": allocation.crossbars_used,
        "crossbars_available": allocation.crossbars_available,
        "initial_bottleneck_seconds": allocation.initial_bottleneck_seconds,
        "final_bottleneck_seconds": allocation.final_bottleneck_seconds,
        "stages": {
            stage_id: {
                "replicas": item.replicas,
                "predicted_seconds": item.predicted_seconds,
                "useful_replica_limit": item.useful_replica_limit,
                "crossbars_per_replica": item.crossbars_per_replica,
            }
            for stage_id, item in allocation.allocations.items()
        },
    }
    (output_dir / "allocation.json").write_text(
        json.dumps(allocation_summary, indent=2), encoding="utf-8"
    )
    hardware_summary = dataflow.summary()
    hardware_summary.update(
        {
            "pipeline_events": len(scheduler.events),
            "pipeline_makespan_seconds": (
                scheduler.events[-1].end_seconds if scheduler.events else 0.0
            ),
        }
    )
    (output_dir / "hardware_writes.json").write_text(
        json.dumps(hardware_summary, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), output_dir / "model.pt")


def _evaluate_node_dataset(graph_dataset, model, split: str, device):
    indices = graph_dataset.splits[split].to(device)
    graph = graph_dataset.graph.to(device)
    model.eval()
    with torch.no_grad():
        output = model(graph)
    labels = graph.y[indices]
    predictions = output[indices]
    if "multiclass" in graph_dataset.task_type:
        predictions = predictions.argmax(dim=-1, keepdim=True)
    else:
        predictions = predictions.sigmoid()
    result = graph_dataset.evaluator.eval(
        {
            "y_true": labels.detach().cpu().numpy(),
            "y_pred": predictions.detach().cpu().numpy(),
        }
    )
    return {key: float(value) for key, value in result.items()}


def _train_node_dataset(
    args,
    graph_dataset,
    predictor,
    allocation,
    output_dir: Path,
    config: HardwareConfig,
    device,
) -> None:
    graph = graph_dataset.graph.to(device)
    model = GOPIMNodeGCN(
        args.model_config["input"],
        args.model_config["hidden"],
        args.model_config["output"],
        args.model_config["layers"],
        args.model_config["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.model_config["learning_rate"])
    dataflow = GOPIMDataflow(config)
    scheduler = PipelineScheduler()
    train_indices = graph_dataset.splits["train"].to(device)
    isu_plan = build_isu_plan(
        _edge_array(graph),
        int(graph.num_nodes),
        config.total_crossbars,
        sparse_threshold=config.sparse_average_degree_threshold,
        sparse_fraction=config.sparse_update_fraction,
        dense_fraction=config.dense_update_fraction,
        low_degree_update_period=config.low_degree_update_period,
    )
    update_policy = getattr(args, "update_policy", "degree")
    if update_policy not in {"degree", "loss-aware"}:
        raise ValueError(f"Unsupported node update policy: {update_policy}")
    previous_node_losses = None
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "dataset": graph_dataset.name,
                "seed": args.seed,
                "epochs": args.epochs,
                "model": args.model_config,
                "isu_enabled": not getattr(args, "disable_isu", False),
                "update_policy": (
                    "full-update"
                    if getattr(args, "disable_isu", False)
                    else update_policy
                ),
                "loss_aware_loss_weight": 0.75,
                "average_degree": isu_plan.average_degree,
                "sparse_average_degree_threshold": config.sparse_average_degree_threshold,
                "selected_update_fraction": isu_plan.update_fraction,
                "selected_vertices": len(isu_plan.selected_vertices),
                "low_degree_update_period": config.low_degree_update_period,
                "adaptive_policy": {
                    "sparse_graph_update_fraction": config.sparse_update_fraction,
                    "dense_graph_update_fraction": config.dense_update_fraction,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    stages = _stages_for_graph(graph, model.layer_dimensions)
    stage_predictions = predictor.predict_stages(stages)
    stage_durations = [
        (
            stage.stage_id,
            stage_predictions[stage.stage_id]
            / allocation.allocations[stage.stage_id].replicas,
        )
        for stage in stages
    ]

    with (output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "validation",
                "test",
                "average_degree",
                "selected_vertices",
                "vertices_updated",
                "update_fraction",
            ]
        )
        for epoch in range(args.epochs):
            gradient_mask = None
            if not getattr(args, "disable_isu", False):
                if update_policy == "loss-aware":
                    update_mask = loss_aware_update_mask(
                        isu_plan, epoch, previous_node_losses
                    )
                else:
                    update_mask = np.fromiter(
                        (
                            isu_plan.should_update(vertex_id, epoch)
                            for vertex_id in range(int(graph.num_nodes))
                        ),
                        dtype=bool,
                        count=int(graph.num_nodes),
                    )
                gradient_mask = torch.as_tensor(
                    update_mask, dtype=torch.bool, device=device
                )
            vertices_updated = (
                int(gradient_mask.sum().item())
                if gradient_mask is not None
                else int(graph.num_nodes)
            )
            optimizer.zero_grad(set_to_none=True)
            model.train()
            output = model(graph, dataflow=dataflow, gradient_mask=gradient_mask)
            loss = _loss(
                output[train_indices],
                graph.y[train_indices],
                graph_dataset.task_type,
            )
            if update_policy == "loss-aware" and not getattr(
                args, "disable_isu", False
            ):
                if "multiclass" in graph_dataset.task_type:
                    per_node_loss = nn.functional.cross_entropy(
                        output[train_indices].detach(),
                        graph.y[train_indices].reshape(-1).long(),
                        reduction="none",
                    )
                elif "classification" in graph_dataset.task_type:
                    per_node_loss = nn.functional.binary_cross_entropy_with_logits(
                        output[train_indices].detach(),
                        _target_tensor(
                            graph.y[train_indices], output[train_indices]
                        ),
                        reduction="none",
                    ).mean(dim=-1)
                else:
                    per_node_loss = (
                        output[train_indices].detach()
                        - _target_tensor(
                            graph.y[train_indices], output[train_indices]
                        )
                    ).square().mean(dim=-1)
                previous_node_losses = np.full(
                    int(graph.num_nodes), np.nan, dtype=np.float64
                )
                previous_node_losses[
                    train_indices.detach().cpu().numpy()
                ] = per_node_loss.cpu().numpy()
            loss.backward()
            optimizer.step()
            dataflow.record_backward(
                model.layer_dimensions,
                int(gradient_mask.sum().item())
                if gradient_mask is not None
                else int(graph.num_nodes),
                args.model_config["input"],
            )
            scheduler.schedule(epoch, stage_durations)

            if (epoch + 1) % args.eval_every == 0:
                validation = _evaluate_node_dataset(
                    graph_dataset, model, "valid", device
                )
                test = _evaluate_node_dataset(graph_dataset, model, "test", device)
            else:
                validation, test = {}, {}
            writer.writerow(
                [
                    epoch + 1,
                    float(loss.item()),
                    json.dumps(validation, sort_keys=True),
                    json.dumps(test, sort_keys=True),
                    isu_plan.average_degree,
                    len(isu_plan.selected_vertices),
                    vertices_updated,
                    vertices_updated / int(graph.num_nodes),
                ]
            )
            file.flush()
            print(
                f"epoch={epoch + 1} loss={loss.item():.6f} "
                f"validation={validation} test={test}"
            )

    allocation_summary = {
        "crossbars_used": allocation.crossbars_used,
        "crossbars_available": allocation.crossbars_available,
        "initial_bottleneck_seconds": allocation.initial_bottleneck_seconds,
        "final_bottleneck_seconds": allocation.final_bottleneck_seconds,
        "stages": {
            stage_id: {
                "replicas": item.replicas,
                "predicted_seconds": item.predicted_seconds,
                "useful_replica_limit": item.useful_replica_limit,
                "crossbars_per_replica": item.crossbars_per_replica,
            }
            for stage_id, item in allocation.allocations.items()
        },
    }
    (output_dir / "allocation.json").write_text(
        json.dumps(allocation_summary, indent=2), encoding="utf-8"
    )
    hardware_summary = dataflow.summary()
    hardware_summary.update(
        {
            "pipeline_events": len(scheduler.events),
            "pipeline_makespan_seconds": (
                scheduler.events[-1].end_seconds if scheduler.events else 0.0
            ),
        }
    )
    (output_dir / "hardware_writes.json").write_text(
        json.dumps(hardware_summary, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), output_dir / "model.pt")


def _write_pipeline_comparison(
    graph_dataset,
    layer_dimensions: list[tuple[int, int]],
    output_dir: Path,
    *,
    micro_batches: int,
    crossbar_budget: int | None,
) -> list[dict]:
    config = HardwareConfig()
    graph = graph_dataset.graph
    stages = _stages_for_graph(graph, layer_dimensions)
    timing_model = StageTimingModel(config)
    results = compare_allocation_schemes(
        stages,
        config,
        micro_batches=micro_batches,
        crossbar_budget=crossbar_budget,
    )
    result_rows = [
        {
            "scheme": result.scheme,
            "crossbars_used": result.crossbars_used,
            "makespan_seconds": result.makespan_seconds,
            "read_array_energy_joules": result.read_array_energy_joules,
            "write_array_energy_joules": result.write_array_energy_joules,
            "array_dynamic_energy_joules": (
                result.read_array_energy_joules
                + result.write_array_energy_joules
            ),
            "speedup_vs_serial": result.speedup_vs_serial,
            "replicas": result.replicas,
        }
        for result in results
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pipeline_comparison.json").write_text(
        json.dumps(
            {
                "dataset": graph_dataset.name,
                "nodes": int(graph.num_nodes),
                "directed_edges": int(graph.edge_index.size(1)),
                "micro_batches": micro_batches,
                "read_latency_seconds": config.read_latency_seconds,
                "write_latency_seconds": config.write_latency_seconds,
                "latency_scope": (
                    "Array pulse-time lower bound only; includes no sense-amp, "
                    "decoder, interconnect, buffer, or control latency."
                ),
                "energy_model": {
                    "name": "NeuroSim-derived 1T1R array dynamic energy estimate",
                    "scope": (
                        "Crossbar line-capacitance switching for reads and "
                        "cell plus line switching for writes. Excludes "
                        "peripheral circuits, global buffer, interconnect, "
                        "leakage, and idle energy."
                    ),
                    "provenance": {
                        "device_and_pulse_parameters": (
                            "NeuroSim V2.0 device/pulse assumptions represented "
                            "in this reduced-scale model; the upstream source "
                            "tree is not included."
                        ),
                        "technology_supply": (
                            "NeuroSim V2.0 technology supply assumption; the "
                            "upstream source tree is not included."
                        ),
                        "read_array_equation": (
                            "First-order adaptation of the NeuroSim V2.0 "
                            "conventional-parallel read array-capacitance "
                            "equation."
                        ),
                        "write_array_equation": (
                            "First-order adaptation of NeuroSim V2.0 write-cell "
                            "and line-energy equations, simplified to one "
                            "pulse and a fully updated row."
                        ),
                    },
                    "assumptions": {
                        "active_read_row_fraction": config.read_active_row_fraction,
                        "average_cell_resistance_ohms": (
                            config.reram_resistance_on_ohms
                            + config.reram_resistance_off_ohms
                        )
                        / 2,
                        "updated_cells_per_write_operation": config.crossbar_cols,
                        "write_pulses_per_updated_cell": config.write_pulses_per_cell,
                        "columns_per_sense_amp": config.columns_per_sense_amp,
                    },
                    "per_crossbar_read_operation_joules": (
                        timing_model.read_array_energy_per_operation()
                    ),
                    "per_row_write_operation_joules": (
                        timing_model.write_array_energy_per_row()
                    ),
                },
                "slimgnn_like_definition": (
                    "A static proxy that gives each stage the same replica "
                    "count; this is not an exact SlimGNN implementation."
                ),
                "results": result_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "pipeline_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "scheme",
                "crossbars_used",
                "makespan_seconds",
                "read_array_energy_joules",
                "write_array_energy_joules",
                "array_dynamic_energy_joules",
                "speedup_vs_serial",
            ),
        )
        writer.writeheader()
        for row in result_rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    return result_rows


def pipeline_command(args) -> None:
    graph_dataset = load_ogb_dataset(args.dataset, args.data_root)
    if not graph_dataset.is_node_prediction or graph_dataset.graph is None:
        raise ValueError("Pipeline comparison requires a node-prediction graph")
    workload = PAPER_WORKLOADS[graph_dataset.name]
    input_width = int(graph_dataset.graph.x.size(1))
    output_width = int(graph_dataset.num_classes or workload["output"])
    hidden_width = args.hidden or workload["hidden"]
    layer_count = args.layers or workload["layers"]
    widths = (
        [input_width]
        + [hidden_width] * (layer_count - 1)
        + [output_width]
    )
    rows = _write_pipeline_comparison(
        graph_dataset,
        list(zip(widths[:-1], widths[1:])),
        Path(args.output_dir),
        micro_batches=args.micro_batches,
        crossbar_budget=args.crossbar_budget,
    )
    print(json.dumps({"dataset": graph_dataset.name, "results": rows}, indent=2))


def small_graphs_command(args) -> None:
    output_dir = Path(args.output_dir)
    summaries = []
    dataset_names = [name.strip().lower() for name in args.datasets.split(",")]
    if len(dataset_names) != 2 or set(dataset_names) != {"cora", "citeseer"}:
        raise ValueError("small-graphs requires exactly cora,citeseer")
    for dataset_name in dataset_names:
        graph_dataset = load_ogb_dataset(dataset_name, args.data_root)
        graph_results = {}
        for variant, disable_isu, update_policy in (
            ("isu", False, "degree"),
            ("loss-aware-isu", False, "loss-aware"),
            ("full-update", True, "degree"),
        ):
            dataset_output = output_dir / dataset_name / variant
            train_args = argparse.Namespace(
                dataset=dataset_name,
                data_root=args.data_root,
                output_dir=str(dataset_output),
                predictor=None,
                profile_samples=128,
                epochs=args.epochs,
                eval_every=1,
                input_channels=None,
                hidden=args.hidden,
                output_channels=None,
                layers=args.layers,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                link_steps_per_epoch=None,
                link_edge_batch_size=8192,
                disable_isu=disable_isu,
                update_policy=update_policy,
                seed=args.seed,
                cpu_threads=args.cpu_threads,
                device=args.device,
            )
            train_command(train_args)
            run_config = json.loads(
                (dataset_output / "run_config.json").read_text(encoding="utf-8")
            )
            with (dataset_output / "metrics.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                metrics = list(csv.DictReader(file))
            if len(metrics) != args.epochs:
                raise RuntimeError(
                    f"{dataset_name}/{variant} completed {len(metrics)} of "
                    f"{args.epochs} requested epochs"
                )
            last_metrics = metrics[-1]
            graph_results[variant] = {
                "run_config": run_config,
                "final_validation_metrics": json.loads(last_metrics["validation"]),
                "final_test_metrics": json.loads(last_metrics["test"]),
                "total_vertex_update_count": sum(
                    int(row["vertices_updated"]) for row in metrics
                ),
                "full_update_count": int(graph_dataset.graph.num_nodes)
                * args.epochs,
                "final_epoch_vertex_update_count": int(
                    last_metrics["vertices_updated"]
                ),
                "final_epoch_update_fraction": float(
                    last_metrics["update_fraction"]
                ),
            }
        isu_result = graph_results["isu"]
        loss_aware_result = graph_results["loss-aware-isu"]
        full_result = graph_results["full-update"]
        model_config = isu_result["run_config"]["model"]
        widths = (
            [model_config["input"]]
            + [model_config["hidden"]] * (model_config["layers"] - 1)
            + [model_config["output"]]
        )
        pipeline_rows = _write_pipeline_comparison(
            graph_dataset,
            list(zip(widths[:-1], widths[1:])),
            output_dir / dataset_name / "pipeline",
            micro_batches=args.micro_batches,
            crossbar_budget=args.crossbar_budget,
        )
        validation_metrics = isu_result["final_validation_metrics"]
        test_metrics = isu_result["final_test_metrics"]
        validation_delta = {
            key: value - full_result["final_validation_metrics"][key]
            for key, value in validation_metrics.items()
            if key in full_result["final_validation_metrics"]
        }
        loss_aware_validation_delta = {
            key: value - full_result["final_validation_metrics"][key]
            for key, value in loss_aware_result["final_validation_metrics"].items()
            if key in full_result["final_validation_metrics"]
        }
        loss_aware_vs_degree_validation_delta = {
            key: value - validation_metrics[key]
            for key, value in loss_aware_result["final_validation_metrics"].items()
            if key in validation_metrics
        }
        test_delta = {
            key: value - full_result["final_test_metrics"][key]
            for key, value in test_metrics.items()
            if key in full_result["final_test_metrics"]
        }
        loss_aware_test_delta = {
            key: value - full_result["final_test_metrics"][key]
            for key, value in loss_aware_result["final_test_metrics"].items()
            if key in full_result["final_test_metrics"]
        }
        loss_aware_vs_degree_test_delta = {
            key: value - isu_result["final_test_metrics"][key]
            for key, value in loss_aware_result["final_test_metrics"].items()
            if key in isu_result["final_test_metrics"]
        }
        summaries.append(
            {
                "dataset": dataset_name,
                "epochs": args.epochs,
                "seed": args.seed,
                "model": model_config,
                "average_degree": isu_result["run_config"]["average_degree"],
                "adaptive_update_fraction": (
                    isu_result["run_config"]["selected_update_fraction"]
                ),
                "selected_vertices": isu_result["run_config"]["selected_vertices"],
                "isu": isu_result,
                "loss_aware_isu": loss_aware_result,
                "full_update": full_result,
                "isu_minus_full_update_validation": validation_delta,
                "isu_minus_full_update_test": test_delta,
                "loss_aware_isu_minus_full_update_validation": (
                    loss_aware_validation_delta
                ),
                "loss_aware_isu_minus_degree_isu_validation": (
                    loss_aware_vs_degree_validation_delta
                ),
                "loss_aware_isu_minus_full_update_test": loss_aware_test_delta,
                "loss_aware_isu_minus_degree_isu_test": (
                    loss_aware_vs_degree_test_delta
                ),
                "pipeline_comparison": pipeline_rows,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "update_comparison.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    with (output_dir / "update_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "dataset",
                "epochs",
                "seed",
                "average_degree",
                "isu_update_fraction",
                "isu_total_vertex_updates",
                "loss_aware_isu_total_vertex_updates",
                "full_update_total_vertex_updates",
                "isu_final_validation",
                "loss_aware_isu_final_validation",
                "full_update_final_validation",
                "isu_final_test",
                "loss_aware_isu_final_test",
                "full_update_final_test",
                "isu_minus_full_update_test",
                "loss_aware_isu_minus_full_update_test",
                "loss_aware_isu_minus_degree_isu_test",
            ]
        )
        for result in summaries:
            writer.writerow(
                [
                    result["dataset"],
                    result["epochs"],
                    result["seed"],
                    result["average_degree"],
                    result["adaptive_update_fraction"],
                    result["isu"]["total_vertex_update_count"],
                    result["loss_aware_isu"]["total_vertex_update_count"],
                    result["full_update"]["total_vertex_update_count"],
                    json.dumps(
                        result["isu"]["final_validation_metrics"], sort_keys=True
                    ),
                    json.dumps(
                        result["loss_aware_isu"]["final_validation_metrics"],
                        sort_keys=True,
                    ),
                    json.dumps(
                        result["full_update"]["final_validation_metrics"],
                        sort_keys=True,
                    ),
                    json.dumps(result["isu"]["final_test_metrics"], sort_keys=True),
                    json.dumps(
                        result["loss_aware_isu"]["final_test_metrics"],
                        sort_keys=True,
                    ),
                    json.dumps(
                        result["full_update"]["final_test_metrics"], sort_keys=True
                    ),
                    json.dumps(
                        result["isu_minus_full_update_test"], sort_keys=True
                    ),
                    json.dumps(
                        result["loss_aware_isu_minus_full_update_test"],
                        sort_keys=True,
                    ),
                    json.dumps(
                        result["loss_aware_isu_minus_degree_isu_test"],
                        sort_keys=True,
                    ),
                ]
            )
    print(json.dumps(summaries, indent=2))


def train_command(args) -> None:
    try:
        from torch_geometric.loader import DataLoader
        from torch.utils.data import Subset
    except ImportError as error:
        raise RuntimeError(
            "The OGB training command requires torch-geometric."
        ) from error

    config = HardwareConfig()
    if args.device == "cpu" or (
        args.device == "auto" and not torch.cuda.is_available()
    ):
        torch.set_num_threads(getattr(args, "cpu_threads", 4))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    graph_dataset = load_ogb_dataset(args.dataset, args.data_root)
    workload = PAPER_WORKLOADS.get(
        graph_dataset.name,
        {
            "input": max(1, graph_dataset.num_node_features),
            "hidden": 64,
            "output": max(1, graph_dataset.num_tasks),
            "layers": 2,
            "learning_rate": 1e-3,
            "dropout": 0.5,
        },
    )
    args.model_config = {
        **workload,
        "input": args.input_channels
        or (
            graph_dataset.num_node_features
            if graph_dataset.is_node_prediction
            else workload["input"]
        ),
        "hidden": args.hidden or workload["hidden"],
        "output": args.output_channels
        or (
            graph_dataset.num_classes
            if graph_dataset.is_node_prediction and graph_dataset.num_classes
            else workload["output"]
        ),
        "layers": args.layers or workload["layers"],
        "learning_rate": args.learning_rate or workload["learning_rate"],
        "dropout": workload["dropout"] if args.dropout is None else args.dropout,
    }
    if graph_dataset.is_node_prediction:
        widths = (
            [args.model_config["input"]]
            + [args.model_config["hidden"]] * (args.model_config["layers"] - 1)
            + [args.model_config["output"]]
        )
    else:
        widths = [args.model_config["input"]] + [
            args.model_config["hidden"]
        ] * args.model_config["layers"]
    layer_dimensions = list(zip(widths[:-1], widths[1:]))
    output_dir = Path(args.output_dir)
    predictor = _fit_or_load_predictor(
        graph_dataset,
        layer_dimensions,
        config,
        output_dir,
        args.profile_samples,
        Path(args.predictor) if args.predictor else None,
    )

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if graph_dataset.is_link_prediction:
        reference_graph = graph_dataset.graph
        reference_stages = _stages_for_graph(reference_graph, layer_dimensions)
        allocation = allocate_replicas(
            reference_stages,
            predictor.predict_stages(reference_stages),
            config,
        )
        _train_link_dataset(
            args,
            graph_dataset,
            predictor,
            allocation,
            output_dir,
            config,
            device,
        )
        return
    if graph_dataset.is_node_prediction:
        reference_graph = graph_dataset.graph
        reference_stages = _stages_for_graph(reference_graph, layer_dimensions)
        allocation = allocate_replicas(
            reference_stages,
            predictor.predict_stages(reference_stages),
            config,
        )
        _train_node_dataset(
            args,
            graph_dataset,
            predictor,
            allocation,
            output_dir,
            config,
            device,
        )
        return

    input_width = args.model_config["input"]
    reference_indices = graph_dataset.splits["train"][: config.micro_batch_size].tolist()
    reference_loader = DataLoader(
        Subset(graph_dataset.dataset, reference_indices),
        batch_size=config.micro_batch_size,
        shuffle=False,
    )
    reference_graph = next(iter(reference_loader))
    reference_stages = _stages_for_graph(reference_graph, layer_dimensions)
    allocation = allocate_replicas(
        reference_stages,
        predictor.predict_stages(reference_stages),
        config,
    )
    model = GOPIMGCN(
        input_width,
        args.model_config["hidden"],
        graph_dataset.num_tasks,
        layer_count=args.model_config["layers"],
        dropout=args.model_config["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.model_config["learning_rate"]
    )
    scheduler = PipelineScheduler()
    dataflow = GOPIMDataflow(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "validation", "test"])
        for epoch in range(args.epochs):
            train_loss = _train_epoch(
                args,
                graph_dataset,
                model,
                optimizer,
                predictor,
                allocation,
                scheduler,
                dataflow,
                config,
                epoch,
                device,
            )
            if (epoch + 1) % args.eval_every == 0:
                validation = _evaluate(
                    graph_dataset, model, "valid", config.micro_batch_size, device
                )
                test = _evaluate(
                    graph_dataset, model, "test", config.micro_batch_size, device
                )
            else:
                validation, test = {}, {}
            writer.writerow(
                [
                    epoch + 1,
                    train_loss,
                    json.dumps(validation, sort_keys=True),
                    json.dumps(test, sort_keys=True),
                ]
            )
            file.flush()
            print(
                f"epoch={epoch + 1} loss={train_loss:.6f} "
                f"validation={validation} test={test}"
            )

    allocation_summary = {
        "crossbars_used": allocation.crossbars_used,
        "crossbars_available": allocation.crossbars_available,
        "initial_bottleneck_seconds": allocation.initial_bottleneck_seconds,
        "final_bottleneck_seconds": allocation.final_bottleneck_seconds,
        "stages": {
            stage_id: {
                "replicas": item.replicas,
                "predicted_seconds": item.predicted_seconds,
                "useful_replica_limit": item.useful_replica_limit,
                "crossbars_per_replica": item.crossbars_per_replica,
            }
            for stage_id, item in allocation.allocations.items()
        },
    }
    (output_dir / "allocation.json").write_text(
        json.dumps(allocation_summary, indent=2), encoding="utf-8"
    )
    hardware_summary = dataflow.summary()
    hardware_summary.update(
        {
            "pipeline_events": len(scheduler.events),
            "pipeline_makespan_seconds": (
                scheduler.events[-1].end_seconds if scheduler.events else 0.0
            ),
        }
    )
    (output_dir / "hardware_writes.json").write_text(
        json.dumps(
            hardware_summary,
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), output_dir / "model.pt")


def isu_command(args) -> None:
    config = HardwareConfig()
    graph_dataset = load_ogb_dataset(args.dataset, args.data_root)
    graph = (
        graph_dataset.graph
        if graph_dataset.is_link_prediction or graph_dataset.is_node_prediction
        else graph_dataset.dataset[int(args.graph_index)]
    )
    result = _write_isu_mapping(Path(args.output), graph, config)
    print(
        json.dumps(
            {
                "vertices": len(result.degrees),
                "average_degree": result.average_degree,
                "update_fraction": result.update_fraction,
                "selected_vertices": len(result.selected_vertices),
                "mapped_crossbars": len(result.assignments),
                "mapping": str(Path(args.output).resolve()),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gopim")
    commands = parser.add_subparsers(dest="command", required=True)

    profile_parser = commands.add_parser(
        "profile", help="Profile OGB graphs and train the stage-time MLP"
    )
    profile_parser.add_argument(
        "--datasets",
        default="ddi,collab,ppa,proteins,arxiv,products",
        help="Paper's six OGB workloads: ddi, collab, ppa, proteins, arxiv, products",
    )
    profile_parser.add_argument("--data-root", default="data")
    profile_parser.add_argument("--output-dir", default="runs/profile")
    profile_parser.add_argument("--samples", type=int, default=128)
    profile_parser.add_argument("--hidden", type=int)
    profile_parser.add_argument("--mlp-epochs", type=int, default=150)
    profile_parser.add_argument(
        "--measured-profiles",
        help="CSV of timings measured from a simulator, with the documented 10 feature columns",
    )
    profile_parser.set_defaults(function=profile_command)

    train_parser = commands.add_parser("train", help="Train a GOPIM GCN on OGB")
    train_parser.add_argument("--dataset", default="ddi")
    train_parser.add_argument("--data-root", default="data")
    train_parser.add_argument("--output-dir", default="runs/gopim")
    train_parser.add_argument("--predictor")
    train_parser.add_argument("--profile-samples", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--eval-every", type=int, default=1)
    train_parser.add_argument("--input-channels", type=int)
    train_parser.add_argument("--hidden", type=int)
    train_parser.add_argument("--output-channels", type=int)
    train_parser.add_argument("--layers", type=int)
    train_parser.add_argument("--dropout", type=float)
    train_parser.add_argument("--learning-rate", type=float)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--cpu-threads", type=int, default=4)
    train_parser.add_argument(
        "--link-steps-per-epoch",
        type=int,
        help="Smoke-test cap; omit to train on the complete link training split",
    )
    train_parser.add_argument(
        "--link-edge-batch-size",
        type=int,
        default=8192,
        help="Number of positive and negative link examples scored per memory-bounded chunk",
    )
    train_parser.add_argument(
        "--disable-isu",
        action="store_true",
        help="Update every vertex each epoch (vanilla control for accuracy comparison)",
    )
    train_parser.add_argument("--device", default="auto")
    train_parser.set_defaults(function=train_command)

    isu_parser = commands.add_parser("isu", help="Export one graph's ISU mapping")
    isu_parser.add_argument("--dataset", default="ddi")
    isu_parser.add_argument("--graph-index", type=int, default=0)
    isu_parser.add_argument("--data-root", default="data")
    isu_parser.add_argument("--output", default="runs/isu_mapping.csv")
    isu_parser.set_defaults(function=isu_command)

    small_graphs_parser = commands.add_parser(
        "small-graphs",
        help="Train selectively on Cora and CiteSeer and compare pipeline allocators",
    )
    small_graphs_parser.add_argument("--datasets", default="cora,citeseer")
    small_graphs_parser.add_argument("--data-root", default="data")
    small_graphs_parser.add_argument("--output-dir", default="runs/small-graphs")
    small_graphs_parser.add_argument("--epochs", type=int, default=50)
    small_graphs_parser.add_argument("--hidden", type=int)
    small_graphs_parser.add_argument("--layers", type=int)
    small_graphs_parser.add_argument("--dropout", type=float)
    small_graphs_parser.add_argument("--learning-rate", type=float)
    small_graphs_parser.add_argument("--seed", type=int, default=42)
    small_graphs_parser.add_argument("--cpu-threads", type=int, default=4)
    small_graphs_parser.add_argument("--device", default="auto")
    small_graphs_parser.add_argument("--micro-batches", type=int, default=4)
    small_graphs_parser.add_argument("--crossbar-budget", type=int)
    small_graphs_parser.set_defaults(function=small_graphs_command)

    pipeline_parser = commands.add_parser(
        "pipeline",
        help="Compare serial, static SlimGNN-like, and GOPIM allocation models",
    )
    pipeline_parser.add_argument("--dataset", choices=("cora", "citeseer"), default="cora")
    pipeline_parser.add_argument("--data-root", default="data")
    pipeline_parser.add_argument("--output-dir", default="runs/pipeline-comparison")
    pipeline_parser.add_argument("--hidden", type=int)
    pipeline_parser.add_argument("--layers", type=int)
    pipeline_parser.add_argument("--micro-batches", type=int, default=4)
    pipeline_parser.add_argument("--crossbar-budget", type=int)
    pipeline_parser.set_defaults(function=pipeline_command)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args, "samples", 1) <= 0:
        raise ValueError("samples must be positive")
    if getattr(args, "profile_samples", 1) <= 0:
        raise ValueError("profile_samples must be positive")
    if (
        getattr(args, "epochs", 1) <= 0
        or getattr(args, "eval_every", 1) <= 0
        or (
            getattr(args, "link_steps_per_epoch", None) is not None
            and args.link_steps_per_epoch <= 0
        )
        or getattr(args, "link_edge_batch_size", 1) <= 0
        or getattr(args, "cpu_threads", 1) <= 0
    ):
        raise ValueError(
            "epochs, eval_every, link_steps_per_epoch, link_edge_batch_size, and "
            "cpu_threads "
            "must be positive"
        )
    if (
        getattr(args, "micro_batches", 1) <= 0
        or getattr(args, "crossbar_budget", None) is not None
        and args.crossbar_budget <= 0
    ):
        raise ValueError(
            "micro_batches and crossbar_budget must be positive"
        )
    args.function(args)
