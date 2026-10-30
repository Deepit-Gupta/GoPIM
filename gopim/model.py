import torch
from torch import nn


def node_features(batch) -> torch.Tensor:
    if getattr(batch, "x", None) is not None:
        return batch.x.float()
    edge_index = batch.edge_index
    num_nodes = int(batch.num_nodes)
    degree = torch.bincount(edge_index[0], minlength=num_nodes).float()
    return degree.clamp_min(1).log1p().unsqueeze(1)


def _normalized_aggregate(
    values: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    num_nodes = values.size(0)
    self_loops = torch.arange(num_nodes, device=values.device)
    edges = torch.cat((edge_index, torch.stack((self_loops, self_loops))), dim=1)
    source, destination = edges[0], edges[1]
    degree = torch.bincount(destination, minlength=num_nodes).to(values.dtype)
    inv_sqrt_degree = degree.clamp_min(1).rsqrt()
    normalization = inv_sqrt_degree[source] * inv_sqrt_degree[destination]
    messages = values[source] * normalization.unsqueeze(1)
    output = values.new_zeros(values.shape)
    output.index_add_(0, destination, messages)
    return output


class GraphConvolution(nn.Module):
    def __init__(self, input_width: int, output_width: int):
        super().__init__()
        self.linear = nn.Linear(input_width, output_width, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_width))

    def forward(
        self,
        values: torch.Tensor,
        edge_index: torch.Tensor,
        dataflow=None,
    ) -> torch.Tensor:
        if dataflow is None:
            combination = self.linear(values)
            aggregation = _normalized_aggregate(combination, edge_index)
        else:
            combination = dataflow.run_combination(values, self.linear.weight)
            aggregation = dataflow.run_aggregation(
                combination,
                edge_index,
                lambda transformed: _normalized_aggregate(transformed, edge_index),
            )
        return aggregation + self.bias


class GOPIMGCN(nn.Module):
    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        output_tasks: int,
        layer_count: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        if min(input_width, hidden_width, output_tasks, layer_count) <= 0:
            raise ValueError("GCN dimensions and layer_count must be positive")
        widths = [input_width] + [hidden_width] * layer_count
        self.convolutions = nn.ModuleList(
            GraphConvolution(widths[index], widths[index + 1])
            for index in range(layer_count)
        )
        self.classifier = nn.Linear(hidden_width, output_tasks)
        self.dropout = dropout
        self.layer_dimensions = list(zip(widths[:-1], widths[1:]))

    def forward(
        self, batch, dataflow=None, gradient_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        values = node_features(batch)
        for layer_index, convolution in enumerate(self.convolutions):
            values = convolution(values, batch.edge_index, dataflow)
            values = (
                dataflow.activate(values)
                if dataflow is not None
                else torch.relu(values)
            )
            if gradient_mask is not None:
                values = _mask_vertex_gradient(values, gradient_mask)
            if layer_index + 1 < len(self.convolutions):
                values = torch.dropout(values, self.dropout, self.training)

        graph_ids = getattr(batch, "batch", None)
        if graph_ids is None:
            graph_ids = torch.zeros(values.size(0), dtype=torch.long, device=values.device)
        graph_count = int(getattr(batch, "num_graphs", graph_ids.max().item() + 1))
        pooled = values.new_zeros((graph_count, values.size(1)))
        pooled.index_add_(0, graph_ids, values)
        counts = torch.bincount(graph_ids, minlength=graph_count).to(values.dtype)
        pooled = pooled / counts.clamp_min(1).unsqueeze(1)
        return self.classifier(pooled)


class GOPIMLinkGCN(nn.Module):
    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        layer_count: int = 2,
        dropout: float = 0.0,
        num_nodes: int | None = None,
    ):
        super().__init__()
        if min(input_width, hidden_width, layer_count) <= 0:
            raise ValueError("GCN dimensions and layer_count must be positive")
        widths = [input_width] + [hidden_width] * layer_count
        self.convolutions = nn.ModuleList(
            GraphConvolution(widths[index], widths[index + 1])
            for index in range(layer_count)
        )
        self.layer_dimensions = list(zip(widths[:-1], widths[1:]))
        self.dropout = dropout
        self.node_embedding = (
            nn.Embedding(num_nodes, input_width) if num_nodes is not None else None
        )
        if self.node_embedding is not None:
            nn.init.xavier_uniform_(self.node_embedding.weight)

    def forward(
        self,
        graph,
        dataflow=None,
        gradient_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if graph.x is not None:
            values = graph.x.float()
        elif self.node_embedding is not None:
            values = self.node_embedding.weight
        else:
            values = node_features(graph)
        if gradient_mask is not None:
            values = _mask_vertex_gradient(values, gradient_mask)
        for index, convolution in enumerate(self.convolutions):
            values = convolution(values, graph.edge_index, dataflow)
            values = (
                dataflow.activate(values)
                if dataflow is not None
                else torch.relu(values)
            )
            if gradient_mask is not None:
                values = _mask_vertex_gradient(values, gradient_mask)
            if index + 1 < len(self.convolutions):
                values = torch.dropout(values, self.dropout, self.training)
        return values


class GOPIMNodeGCN(nn.Module):
    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        output_width: int,
        layer_count: int,
        dropout: float,
    ):
        super().__init__()
        if min(input_width, hidden_width, output_width, layer_count) <= 0:
            raise ValueError("GCN dimensions and layer_count must be positive")
        widths = [input_width] + [hidden_width] * (layer_count - 1)
        self.convolutions = nn.ModuleList(
            GraphConvolution(widths[index], widths[index + 1])
            for index in range(layer_count - 1)
        )
        self.classifier = GraphConvolution(widths[-1], output_width)
        self.layer_dimensions = list(zip(widths[:-1], widths[1:])) + [
            (widths[-1], output_width)
        ]
        self.dropout = dropout

    def forward(
        self,
        graph,
        dataflow=None,
        gradient_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = node_features(graph)
        for convolution in self.convolutions:
            values = convolution(values, graph.edge_index, dataflow)
            values = (
                dataflow.activate(values)
                if dataflow is not None
                else torch.relu(values)
            )
            if gradient_mask is not None:
                values = _mask_vertex_gradient(values, gradient_mask)
            values = torch.dropout(values, self.dropout, self.training)
        values = self.classifier(values, graph.edge_index, dataflow)
        if gradient_mask is not None:
            values = _mask_vertex_gradient(values, gradient_mask)
        return values


def _mask_vertex_gradient(
    values: torch.Tensor, gradient_mask: torch.Tensor
) -> torch.Tensor:
    if gradient_mask.ndim != 1 or gradient_mask.numel() != values.size(0):
        raise ValueError("gradient_mask must contain one value per vertex")
    if not values.requires_grad:
        return values
    mask = gradient_mask.to(device=values.device, dtype=values.dtype).reshape(-1, 1)
    values.register_hook(lambda gradient: gradient * mask)
    return values
