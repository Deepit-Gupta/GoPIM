import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from gopim.allocator import allocate_replicas
from gopim.benchmark import compare_allocation_schemes
from gopim import cli
from gopim.config import HardwareConfig
from gopim.hardware import GlobalBuffer, GOPIMDataflow, StageTimingModel
from gopim.isu import build_isu_plan, loss_aware_update_mask
from gopim.model import GOPIMGCN
from gopim.model import GOPIMLinkGCN, GOPIMNodeGCN
from gopim.pipeline import PipelineScheduler
from gopim.predictor import ProfileRecord, StageTimePredictor
from gopim.stages import four_stages_per_layer


class HardwareConfigTests(unittest.TestCase):
    def test_default_crossbar_capacity_matches_16_gib(self):
        config = HardwareConfig()
        self.assertEqual(config.crossbars_per_tile, 256)
        self.assertEqual(config.total_crossbars, 16_777_216)
        self.assertEqual(config.crossbar_capacity_bytes, 16 * 1024**3)
        config.validate_resource_capacity()

    def test_global_buffer_counts_streamed_transfers(self):
        buffer = GlobalBuffer(HardwareConfig(global_buffer_bytes=1024))
        self.assertEqual(buffer.transfer(2049), 3)
        buffer.transfer(12, write=True)
        self.assertEqual(buffer.read_bytes, 2049)
        self.assertEqual(buffer.write_bytes, 12)


class StageTests(unittest.TestCase):
    def test_four_stages_per_layer_and_forward_backward_order(self):
        stages = four_stages_per_layer([(16, 32), (32, 64)], 200, 400)
        self.assertEqual(len(stages), 8)
        self.assertEqual(
            [stage.stage_id for stage in stages[:4]],
            [
                "layer-0-CO-forward",
                "layer-0-AG-forward",
                "layer-1-CO-forward",
                "layer-1-AG-forward",
            ],
        )
        self.assertEqual(stages[4].stage_id, "layer-1-LC-backward")
        self.assertEqual(stages[7].stage_id, "layer-0-GC-backward")
        self.assertEqual([stage.phase for stage in stages], [
            "CO", "AG", "CO", "AG", "LC", "GC", "LC", "GC"
        ])

    def test_timing_decreases_with_replicas(self):
        config = HardwareConfig()
        stage = four_stages_per_layer([(128, 128)], 256, 2000)[0]
        model = StageTimingModel(config)
        self.assertGreater(
            model.estimate(stage, replicas=1).seconds,
            model.estimate(stage, replicas=4).seconds,
        )

    def test_allocator_improves_bottleneck_with_two_heaps(self):
        stages = four_stages_per_layer([(64, 64)], 64, 100)
        predictions = {
            stage.stage_id: (10.0 if stage.stage_id.endswith("CO-forward") else 3.0)
            for stage in stages
        }
        config = HardwareConfig(tile_count=1, pes_per_tile=1)
        result = allocate_replicas(stages, predictions, config)
        self.assertGreater(result.crossbars_used, len(stages))
        self.assertEqual(result.crossbars_used, result.crossbars_available)
        self.assertLess(result.final_bottleneck_seconds, 10.0)
        self.assertTrue(
            all(item.replicas <= item.useful_replica_limit for item in result.allocations.values())
        )
        self.assertEqual(
            result.crossbars_used,
            sum(
                item.replicas * item.crossbars_per_replica
                for item in result.allocations.values()
            ),
        )


class ISUTests(unittest.TestCase):
    def test_sparse_graph_updates_top_80_percent_and_throttles_rest(self):
        edges = np.array([[0, 0, 0, 1, 1, 2], [1, 2, 3, 0, 2, 0]])
        plan = build_isu_plan(edges, 10, 4)
        self.assertEqual(plan.update_fraction, 0.8)
        self.assertEqual(len(plan.selected_vertices), 8)
        low_degree = next(vertex for vertex in range(10) if vertex not in plan.selected_vertices)
        self.assertFalse(plan.should_update(low_degree, 0))
        self.assertTrue(plan.should_update(low_degree, 19))
        self.assertTrue(plan.should_update(next(iter(plan.selected_vertices)), 0))
        self.assertTrue(all(len(values) == 2 for values in plan.assignments.values()))

    def test_dense_graph_updates_top_50_percent(self):
        source, target = [], []
        for left in range(10):
            for right in range(10):
                if left != right:
                    source.append(left)
                    target.append(right)
        plan = build_isu_plan(np.array([source, target]), 10, 3)
        self.assertGreater(plan.average_degree, 8)
        self.assertEqual(plan.update_fraction, 0.5)
        self.assertEqual(len(plan.selected_vertices), 5)

    def test_loss_aware_updates_reprioritize_at_equal_budget(self):
        source, target = [], []
        for left in range(10):
            for right in range(10):
                if left != right:
                    source.append(left)
                    target.append(right)
        source.extend([0] * 20)
        target.extend([1] * 20)
        plan = build_isu_plan(np.array([source, target]), 10, 4)
        losses = np.zeros(10)
        losses[9] = 10.0
        mask = loss_aware_update_mask(plan, 0, losses)
        self.assertEqual(int(mask.sum()), len(plan.selected_vertices))
        self.assertNotIn(9, plan.selected_vertices)
        self.assertTrue(mask[9])
        self.assertTrue(loss_aware_update_mask(plan, 19, losses).all())


class PredictorAndPipelineTests(unittest.TestCase):
    def test_mlp_fit_predict_save_and_load(self):
        records = [
            ProfileRecord(
                tuple(float(i + j) for j in range(10)),
                1e-6 * (i + 1),
                "unit",
                str(i),
            )
            for i in range(1, 5)
        ]
        predictor = StageTimePredictor()
        predictor.fit(records, epochs=3)
        prediction = predictor.predict(records[0].features)
        self.assertGreater(prediction, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictor.pt"
            predictor.save(path)
            loaded = StageTimePredictor.load(path)
            self.assertAlmostEqual(prediction, loaded.predict(records[0].features), places=10)

    def test_profile_csv_loads_simulator_measurements(self):
        from gopim.predictor import load_profile_csv

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measured.csv"
            path.write_text(
                "dataset,stage_id,duration_seconds,R_IFM_CO,C_IFM_CO,R_W_CO,C_W_CO,"
                "R_A_AG,C_A_AG,R_F_AG,C_F_AG,sparsity,layer\n"
                "ddi,CO1,0.0001,64,256,256,256,64,64,64,256,0.9,0\n",
                encoding="utf-8",
            )
            records = load_profile_csv(path)
        self.assertEqual(records[0].dataset, "ddi")
        self.assertAlmostEqual(records[0].duration_seconds, 1e-4)
        self.assertEqual(len(records[0].features), 10)

    def test_pipeline_overlaps_adjacent_micro_batches(self):
        scheduler = PipelineScheduler()
        for micro_batch in range(3):
            makespan = scheduler.schedule(micro_batch, [("CO", 1.0), ("AG", 1.0)])
        self.assertEqual(makespan, 4.0)
        self.assertEqual(len(scheduler.events), 6)

    def test_pipeline_schemes_report_latency_energy_and_resource_use(self):
        config = HardwareConfig(tile_count=1, pes_per_tile=1)
        stages = four_stages_per_layer([(64, 64)], 64, 100)
        results = compare_allocation_schemes(
            stages, config, micro_batches=3, crossbar_budget=12
        )
        self.assertEqual(
            [result.scheme for result in results],
            [
                "serial",
                "slimgnn_like_static_equal_replication",
                "gopim_greedy_bottleneck",
            ],
        )
        self.assertEqual(results[0].speedup_vs_serial, 1.0)
        self.assertTrue(
            all(
                result.read_array_energy_joules > 0
                and result.write_array_energy_joules > 0
                for result in results
            )
        )
        self.assertTrue(all(result.crossbars_used <= 12 for result in results))
        self.assertEqual(results[0].crossbars_used, 12)
        self.assertAlmostEqual(
            results[0].read_array_energy_joules,
            results[1].read_array_energy_joules,
        )
        self.assertAlmostEqual(
            results[1].write_array_energy_joules,
            results[2].write_array_energy_joules,
        )

    def test_neurosim_array_energy_uses_source_parameters_and_activity(self):
        config = HardwareConfig()
        stage = four_stages_per_layer([(64, 64)], 64, 100)[-1]
        timing_model = StageTimingModel(config)
        timing = timing_model.estimate(stage)
        line_capacitance = (
            config.crossbar_rows
            * config.cell_height_in_feature_size
            * config.feature_size_meters
            * config.wire_capacitance_farad_per_meter
        )
        expected_read_per_operation = (
            line_capacitance * config.read_voltage**2 * 8
            + line_capacitance
            * config.technology_supply_voltage**2
            * config.crossbar_rows
            * 0.5
        ) * 8
        expected_write_per_row = (
            config.crossbar_cols
            * config.write_voltage**2
            / ((config.reram_resistance_on_ohms + config.reram_resistance_off_ohms) / 2)
            * config.write_latency_seconds
            + line_capacitance
            * config.technology_supply_voltage**2
            + line_capacitance
            * config.write_voltage**2
            * config.crossbar_cols
        )
        self.assertAlmostEqual(
            timing_model.read_array_energy_per_operation(),
            expected_read_per_operation,
        )
        self.assertAlmostEqual(
            timing_model.write_array_energy_per_row(),
            expected_write_per_row,
        )
        self.assertAlmostEqual(
            timing.read_array_energy_joules,
            timing.read_operations
            * timing_model.read_array_energy_per_operation(),
        )
        self.assertAlmostEqual(
            timing.write_array_energy_joules,
            timing.write_operations * timing_model.write_array_energy_per_row(),
        )
        self.assertGreater(timing.write_array_energy_joules, 0)
        lower_activity_config = HardwareConfig(read_active_row_fraction=0.25)
        self.assertLess(
            StageTimingModel(lower_activity_config).read_array_energy_per_operation(),
            timing_model.read_array_energy_per_operation(),
        )

    def test_pipeline_report_records_energy_provenance_and_scope(self):
        graph = Data(
            x=torch.ones((4, 4)),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        )
        graph_dataset = SimpleNamespace(
            graph=graph,
            name="cora",
        )
        with tempfile.TemporaryDirectory() as directory:
            cli._write_pipeline_comparison(
                graph_dataset,
                [(4, 8)],
                Path(directory),
                micro_batches=2,
                crossbar_budget=12,
            )
            report = json.loads(
                (Path(directory) / "pipeline_comparison.json").read_text(
                    encoding="utf-8"
                )
            )
        energy_model = report["energy_model"]
        self.assertIn(
            "first-order adaptation",
            energy_model["provenance"]["read_array_equation"].lower(),
        )
        self.assertIn(
            "first-order adaptation",
            energy_model["provenance"]["write_array_equation"].lower(),
        )
        self.assertIn("excludes", energy_model["scope"].lower())
        self.assertEqual(len(report["results"]), 3)


class ModelTests(unittest.TestCase):
    def test_link_split_supports_source_and_target_negative_format(self):
        positive, negative = cli._link_edges(
            {
                "source_node": torch.tensor([0, 1]),
                "target_node": torch.tensor([2, 3]),
                "target_node_neg": torch.tensor([[4, 5], [5, 6]]),
            },
            torch.device("cpu"),
        )
        self.assertEqual(tuple(positive.shape), (2, 2))
        self.assertEqual(tuple(negative.shape), (2, 2, 2))

    def test_link_evaluation_preserves_grouped_negative_edges(self):
        observed_shapes = []
        graph_dataset = SimpleNamespace(
            splits={
                "valid": {
                    "edge": torch.tensor([[0, 1]]),
                    "edge_neg": torch.tensor([[[0, 2], [0, 3]]]),
                }
            },
            evaluator=SimpleNamespace(
                eval=lambda values: (
                    observed_shapes.append(values["y_pred_neg"].shape) or {"mrr": 1.0}
                )
            ),
        )

        class EmbeddingModel:
            def eval(self):
                return None

            def __call__(self, graph):
                return torch.tensor([[0.0], [1.0], [2.0], [3.0]])

        cli._evaluate_link_dataset(
            graph_dataset, EmbeddingModel(), object(), "valid", torch.device("cpu")
        )
        self.assertEqual(observed_shapes, [(1, 2)])

    def test_gcn_executes_forward_and_backward(self):
        graph_a = Data(
            x=torch.ones((3, 4)),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            y=torch.tensor([1.0]),
        )
        graph_b = Data(
            x=torch.ones((2, 4)),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            y=torch.tensor([0.0]),
        )
        batch = Batch.from_data_list([graph_a, graph_b])
        model = GOPIMGCN(4, 8, 1, layer_count=2)
        output = model(batch)
        self.assertEqual(tuple(output.shape), (2, 1))
        output.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_training_dataflow_records_sram_and_reram_updates(self):
        dataflow = GOPIMDataflow(HardwareConfig())
        features = torch.rand((32, 4), requires_grad=True)
        weights = torch.rand((8, 4), requires_grad=True)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        combined = dataflow.run_combination(features, weights)
        aggregated = dataflow.run_aggregation(
            combined, edge_index, lambda values: values
        )
        dataflow.activate(aggregated).sum().backward()
        dataflow.record_backward([(4, 8), (8, 8)], 20, 4)
        summary = dataflow.summary()
        self.assertEqual(summary["operation_combination_mvm"], 1)
        self.assertEqual(summary["operation_aggregation_mvm"], 1)
        self.assertEqual(summary["operation_weight_gradient_multiplication"], 2)
        self.assertEqual(summary["sram_weight_values_written"], 96)
        self.assertEqual(summary["reram_weight_values_written"], 96)
        self.assertEqual(summary["reram_feature_values_written"], 80)

    def test_link_model_masks_nonselected_vertex_updates(self):
        graph = Data(
            x=None,
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            num_nodes=3,
        )
        model = GOPIMLinkGCN(4, 5, num_nodes=3)
        with torch.no_grad():
            model.node_embedding.weight.fill_(1)
            for convolution in model.convolutions:
                convolution.linear.weight.fill_(0.1)
                convolution.bias.fill_(0.1)
        output = model(graph, gradient_mask=torch.tensor([True, False, True]))
        output.sum().backward()
        self.assertGreater(float(model.node_embedding.weight.grad[0].abs().sum()), 0)
        self.assertEqual(float(model.node_embedding.weight.grad[1].abs().sum()), 0)
        self.assertGreater(float(model.node_embedding.weight.grad[2].abs().sum()), 0)

    def test_node_model_produces_paper_output_dimensions(self):
        graph = Data(
            x=torch.rand((3, 4)),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            y=torch.tensor([[0], [1], [0]]),
        )
        model = GOPIMNodeGCN(4, 8, 3, 3, 0.5)
        self.assertEqual(tuple(model(graph).shape), (3, 3))

    def test_training_command_runs_end_to_end_on_graph_fixtures(self):
        class FixtureDataset:
            def __init__(self):
                self.graphs = [
                    Data(
                        x=torch.rand((4, 4)),
                        edge_index=torch.tensor(
                            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]
                        ),
                        y=torch.tensor([float(index % 2)]),
                    )
                    for index in range(4)
                ]

            def __getitem__(self, index):
                return self.graphs[index]

            def __len__(self):
                return len(self.graphs)

        graph_dataset = SimpleNamespace(
            dataset=FixtureDataset(),
            splits={
                "train": torch.tensor([0, 1]),
                "valid": torch.tensor([2]),
                "test": torch.tensor([3]),
            },
            evaluator=SimpleNamespace(eval=lambda values: {"fixture_metric": 1.0}),
            name="ogbg-ddi",
            task_type="binary classification",
            num_tasks=1,
            num_node_features=4,
            graph=None,
            is_link_prediction=False,
            is_node_prediction=False,
        )
        args = SimpleNamespace(
            dataset="ddi",
            data_root="unused",
            output_dir="",
            predictor=None,
            profile_samples=2,
            epochs=1,
            eval_every=1,
            hidden=8,
            learning_rate=1e-3,
            input_channels=None,
            output_channels=None,
            layers=None,
            dropout=None,
            link_steps_per_epoch=None,
            link_edge_batch_size=1,
            disable_isu=False,
            seed=42,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = directory
            with patch("gopim.cli.load_ogb_dataset", return_value=graph_dataset):
                cli.train_command(args)
            for name in (
                "execution_time_mlp.pt",
                "stage_profiles.csv",
                "metrics.csv",
                "allocation.json",
                "hardware_writes.json",
                "model.pt",
            ):
                self.assertTrue((Path(directory) / name).is_file(), name)

    def test_small_graphs_runs_matched_isu_and_full_update_controls(self):
        graph = Data(
            x=torch.rand((5, 4)),
            edge_index=torch.tensor(
                [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]
            ),
            y=torch.tensor([[0], [1], [0], [1], [0]]),
        )
        graph_dataset = SimpleNamespace(
            graph=graph,
            is_node_prediction=True,
            num_classes=2,
        )
        args = SimpleNamespace(
            datasets="cora,citeseer",
            data_root="unused",
            output_dir="",
            epochs=2,
            hidden=8,
            layers=2,
            dropout=0.0,
            learning_rate=0.01,
            seed=19,
            cpu_threads=1,
            device="cpu",
            micro_batches=2,
            crossbar_budget=None,
        )
        train_calls = []

        def fake_train(train_args):
            train_calls.append(train_args)
            output = Path(train_args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            (output / "run_config.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "input": 4,
                            "hidden": 8,
                            "output": 2,
                            "layers": 2,
                        },
                        "average_degree": 1.2,
                        "selected_update_fraction": 0.8,
                        "selected_vertices": 4,
                    }
                ),
                encoding="utf-8",
            )
            updated = 5 if train_args.disable_isu else 4
            with (output / "metrics.csv").open(
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
                for epoch in range(1, 3):
                    writer.writerow(
                        [
                            epoch,
                            0.5,
                            json.dumps({"acc": 0.6}),
                            json.dumps({"acc": 0.7 if train_args.disable_isu else 0.8}),
                            1.2,
                            4,
                            updated,
                            updated / 5,
                        ]
                    )

        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = directory
            with (
                patch("gopim.cli.load_ogb_dataset", return_value=graph_dataset),
                patch("gopim.cli.train_command", side_effect=fake_train),
                patch("gopim.cli._write_pipeline_comparison", return_value=[]),
            ):
                cli.small_graphs_command(args)

            self.assertEqual(len(train_calls), 6)
            for index in (0, 3):
                degree_call, loss_aware_call, full_call = train_calls[index : index + 3]
                self.assertEqual(degree_call.seed, loss_aware_call.seed)
                self.assertEqual(degree_call.seed, full_call.seed)
                self.assertEqual(degree_call.epochs, full_call.epochs)
                self.assertFalse(degree_call.disable_isu)
                self.assertFalse(loss_aware_call.disable_isu)
                self.assertTrue(full_call.disable_isu)
                self.assertEqual(degree_call.update_policy, "degree")
                self.assertEqual(loss_aware_call.update_policy, "loss-aware")
                self.assertEqual(full_call.update_policy, "degree")
            report = json.loads(
                (Path(directory) / "update_comparison.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(report), 2)
            self.assertAlmostEqual(
                report[0]["isu_minus_full_update_test"]["acc"], 0.1
            )
            self.assertIn("loss_aware_isu", report[0])
            self.assertEqual(
                report[0]["loss_aware_isu"]["total_vertex_update_count"], 8
            )
            with (Path(directory) / "update_comparison.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["isu_total_vertex_updates"], "8")
            self.assertEqual(rows[0]["loss_aware_isu_total_vertex_updates"], "8")
            self.assertEqual(rows[0]["full_update_total_vertex_updates"], "10")

    def test_node_prediction_training_uses_class_count_and_features(self):
        graph = Data(
            x=torch.rand((4, 4)),
            edge_index=torch.tensor(
                [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]
            ),
            y=torch.tensor([[0], [1], [2], [1]]),
        )
        graph_dataset = SimpleNamespace(
            dataset=None,
            graph=graph,
            splits={
                "train": torch.tensor([0, 1]),
                "valid": torch.tensor([2]),
                "test": torch.tensor([3]),
            },
            evaluator=SimpleNamespace(
                eval=lambda values: {
                    "acc": float(
                        (values["y_true"] == values["y_pred"]).mean()
                    )
                }
            ),
            name="ogbn-arxiv",
            task_type="multiclass classification",
            num_tasks=40,
            num_node_features=4,
            num_classes=3,
            is_link_prediction=False,
            is_node_prediction=True,
        )
        args = SimpleNamespace(
            dataset="arxiv",
            data_root="unused",
            output_dir="",
            predictor=None,
            profile_samples=1,
            epochs=1,
            eval_every=1,
            hidden=8,
            learning_rate=None,
            input_channels=None,
            output_channels=None,
            layers=2,
            dropout=0.0,
            link_steps_per_epoch=None,
            seed=42,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = directory
            with patch("gopim.cli.load_ogb_dataset", return_value=graph_dataset):
                cli.train_command(args)
            metrics = Path(directory) / "metrics.csv"
            self.assertTrue(metrics.is_file())
            self.assertTrue((Path(directory) / "model.pt").is_file())
            self.assertTrue((Path(directory) / "run_config.json").is_file())
            with metrics.open(newline="", encoding="utf-8") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["selected_vertices"], "4")
            self.assertEqual(row["vertices_updated"], "4")
            self.assertAlmostEqual(float(row["update_fraction"]), 1.0)

    def test_link_prediction_training_uses_ogb_edge_splits(self):
        graph = Data(
            x=None,
            edge_index=torch.tensor(
                [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]
            ),
            num_nodes=4,
        )
        graph_dataset = SimpleNamespace(
            dataset=None,
            graph=graph,
            splits={
                "train": {"edge": torch.tensor([[0, 1], [1, 2], [2, 3]])},
                "valid": {
                    "edge": torch.tensor([[0, 2]]),
                    "edge_neg": torch.tensor([[0, 3]]),
                },
                "test": {
                    "edge": torch.tensor([[0, 1]]),
                    "edge_neg": torch.tensor([[0, 3]]),
                },
            },
            evaluator=SimpleNamespace(eval=lambda values: {"hits@20": 1.0}),
            name="ogbl-ddi",
            task_type="link prediction",
            num_tasks=1,
            num_node_features=0,
            is_link_prediction=True,
            is_node_prediction=False,
        )
        args = SimpleNamespace(
            dataset="ddi",
            data_root="unused",
            output_dir="",
            predictor=None,
            profile_samples=1,
            epochs=1,
            eval_every=1,
            hidden=8,
            learning_rate=1e-3,
            input_channels=None,
            output_channels=None,
            layers=None,
            dropout=None,
            link_steps_per_epoch=None,
            link_edge_batch_size=1,
            disable_isu=False,
            seed=42,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = directory
            with patch("gopim.cli.load_ogb_dataset", return_value=graph_dataset):
                cli.train_command(args)
            metrics_path = Path(directory) / "metrics.csv"
            self.assertTrue(metrics_path.is_file())
            with metrics_path.open(newline="", encoding="utf-8") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["positive_train_edges"], "3")
            self.assertEqual(row["negative_train_edges"], "3")
            self.assertTrue((Path(directory) / "run_config.json").is_file())
            self.assertTrue((Path(directory) / "model.pt").is_file())


if __name__ == "__main__":
    unittest.main()
