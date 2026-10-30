from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path


ALIASES = {
    "ddi": "ogbl-ddi",
    "collab": "ogbl-collab",
    "ppa": "ogbl-ppa",
    "proteins": "ogbn-proteins",
    "arxiv": "ogbn-arxiv",
    "products": "ogbn-products",
    "cora": "cora",
    "citeseer": "citeseer",
}

PAPER_WORKLOADS = {
    "ogbl-ddi": {"layers": 2, "learning_rate": 0.005, "dropout": 0.5, "input": 256, "hidden": 256, "output": 256},
    "ogbl-collab": {"layers": 3, "learning_rate": 0.001, "dropout": 0.0, "input": 128, "hidden": 256, "output": 256},
    "ogbl-ppa": {"layers": 3, "learning_rate": 0.01, "dropout": 0.0, "input": 58, "hidden": 256, "output": 256},
    "ogbn-proteins": {"layers": 3, "learning_rate": 0.01, "dropout": 0.0, "input": 8, "hidden": 256, "output": 112},
    "ogbn-arxiv": {"layers": 3, "learning_rate": 0.01, "dropout": 0.5, "input": 128, "hidden": 256, "output": 40},
    "ogbn-products": {"layers": 3, "learning_rate": 0.01, "dropout": 0.5, "input": 100, "hidden": 256, "output": 47},
    "cora": {"layers": 2, "learning_rate": 0.01, "dropout": 0.5, "input": 1433, "hidden": 64, "output": 7},
    "citeseer": {"layers": 2, "learning_rate": 0.01, "dropout": 0.5, "input": 3703, "hidden": 64, "output": 6},
}

@dataclass
class OGBGraphDataset:
    dataset: object
    splits: dict
    evaluator: object
    name: str
    task_type: str
    num_tasks: int
    num_node_features: int
    graph: object | None = None
    is_link_prediction: bool = False
    is_node_prediction: bool = False
    num_classes: int | None = None


@contextmanager
def _ogb_torch_load_compat(torch, dataset_root: Path):
    original_load = torch.load
    dataset_root = dataset_root.resolve()
    has_release_marker = (dataset_root / "RELEASE_v1.txt").is_file()

    def load_ogb_file(*args, **kwargs):
        source = kwargs.get("f", args[0] if args else None)
        if not has_release_marker or not isinstance(source, (str, Path)):
            return original_load(*args, **kwargs)
        try:
            Path(source).resolve().relative_to(dataset_root)
        except ValueError:
            return original_load(*args, **kwargs)
        # OGB 1.3.6 caches protocol-4 data; newer PyTorch defaults reject it.
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_ogb_file
    try:
        yield
    finally:
        torch.load = original_load


def load_ogb_dataset(name: str, root: str | Path) -> OGBGraphDataset:
    canonical_name = ALIASES.get(name.lower(), name.lower())
    supported = {
        "ogbl-ddi",
        "ogbl-collab",
        "ogbl-ppa",
        "ogbn-proteins",
        "ogbn-arxiv",
        "ogbn-products",
        "cora",
        "citeseer",
    }
    if canonical_name not in supported:
        raise ValueError(
            f"Unsupported dataset {name!r}; choose ddi, collab, ppa, proteins, "
            "arxiv, products, cora, or citeseer"
        )
    if canonical_name.startswith("ogbl-"):
        try:
            from ogb.linkproppred import Evaluator, LinkPropPredDataset
            import torch
            from torch_geometric.data import Data
        except ImportError as error:
            raise RuntimeError(
                "OGB link prediction support requires `python -m pip install -e .[ogb]`."
            ) from error
        dataset_root = Path(root) / canonical_name.replace("-", "_")
        with _ogb_torch_load_compat(torch, dataset_root):
            dataset = LinkPropPredDataset(name=canonical_name, root=str(root))
            splits = dataset.get_edge_split()
            graph_data = dataset[0]
            node_features = graph_data.get("node_feat")
            node_features_tensor = (
                torch.as_tensor(node_features, dtype=torch.float32)
                if node_features is not None
                else None
            )
            graph = Data(
                x=node_features_tensor,
                edge_index=torch.as_tensor(graph_data["edge_index"], dtype=torch.long),
                num_nodes=int(graph_data["num_nodes"]),
            )
        return OGBGraphDataset(
            dataset=dataset,
            splits=splits,
            evaluator=Evaluator(name=canonical_name),
            name=canonical_name,
            task_type="link prediction",
            num_tasks=1,
            num_node_features=PAPER_WORKLOADS[canonical_name]["input"],
            graph=graph,
            is_link_prediction=True,
        )

    if canonical_name.startswith("ogbn-"):
        try:
            from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
            import torch
        except ImportError as error:
            raise RuntimeError(
                "OGB node prediction support requires `python -m pip install -e .[ogb]`."
            ) from error
        dataset_root = Path(root) / canonical_name.replace("-", "_")
        with _ogb_torch_load_compat(torch, dataset_root):
            dataset = PygNodePropPredDataset(name=canonical_name, root=str(root))
            splits = dataset.get_idx_split()
            graph = dataset[0]
        if canonical_name == "ogbn-proteins" and graph.x is None:
            if getattr(graph, "edge_attr", None) is None:
                raise ValueError("OGBN-Proteins is missing edge attributes for node features")
            from torch_geometric.utils import scatter

            source, destination = graph.edge_index
            node_features = scatter(
                graph.edge_attr.float(),
                destination,
                dim=0,
                dim_size=int(graph.num_nodes),
                reduce="mean",
            )
            graph.x = node_features
        workload = PAPER_WORKLOADS[canonical_name]
        return OGBGraphDataset(
            dataset=dataset,
            splits=splits,
            evaluator=Evaluator(name=canonical_name),
            name=canonical_name,
            task_type=str(dataset.task_type).lower(),
            num_tasks=int(dataset.num_tasks),
            num_node_features=int(graph.x.size(1)) if graph.x is not None else workload["input"],
            graph=graph,
            is_node_prediction=True,
            num_classes=int(workload["output"]),
        )

    if canonical_name in {"cora", "citeseer"}:
        try:
            from torch_geometric.datasets import Planetoid
            import torch
        except ImportError as error:
            raise RuntimeError(
                "Cora support requires OGB and torch-geometric."
            ) from error
        dataset_name = "Cora" if canonical_name == "cora" else "CiteSeer"
        dataset_root = Path(root) / dataset_name
        dataset = Planetoid(root=str(dataset_root), name=dataset_name)
        graph = dataset[0]
        splits = {
            "train": graph.train_mask.nonzero(as_tuple=False).reshape(-1),
            "valid": graph.val_mask.nonzero(as_tuple=False).reshape(-1),
            "test": graph.test_mask.nonzero(as_tuple=False).reshape(-1),
        }

        class CoraEvaluator:
            @staticmethod
            def eval(values):
                true = torch.as_tensor(values["y_true"]).reshape(-1)
                pred = torch.as_tensor(values["y_pred"]).reshape(-1)
                return {"acc": float((true == pred).float().mean().item())}

        return OGBGraphDataset(
            dataset=dataset,
            splits=splits,
            evaluator=CoraEvaluator(),
            name=canonical_name,
            task_type="multiclass classification",
            num_tasks=1,
            num_node_features=int(graph.x.size(1)),
            graph=graph,
            is_node_prediction=True,
            num_classes=int(dataset.num_classes),
        )

    try:
        from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
    except ImportError as error:
        raise RuntimeError(
            "OGB support is optional. Install it with "
            "`python -m pip install -e .[ogb]`."
        ) from error

    import torch

    dataset_root = Path(root) / canonical_name.replace("-", "_")
    with _ogb_torch_load_compat(torch, dataset_root):
        dataset = PygGraphPropPredDataset(name=canonical_name, root=str(root))
        splits = dataset.get_idx_split()
    return OGBGraphDataset(
        dataset=dataset,
        splits=splits,
        evaluator=Evaluator(name=canonical_name),
        name=canonical_name,
        task_type=dataset.task_type.lower(),
        num_tasks=dataset.num_tasks,
        num_node_features=dataset.num_node_features,
        is_link_prediction=False,
    )
