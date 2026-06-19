from __future__ import annotations

import unittest

from rttrainer.export_rtneural.registry import (
    ACTIVATION_SPECS,
    BENCHMARK_SIZES,
    LAYER_SPECS,
    support_matrix,
)


class RTNeuralRegistryTests(unittest.TestCase):
    def test_benchmarked_layers_match_upstream_compare_scope(self) -> None:
        benchmarked = {
            key for key, spec in LAYER_SPECS.items()
            if spec.benchmarked
        }
        self.assertEqual(benchmarked, {"dense", "gru", "lstm", "conv1d"})
        self.assertEqual(BENCHMARK_SIZES, [4, 8, 16, 32, 64])

    def test_benchmarked_activations_match_upstream_compare_scope(self) -> None:
        benchmarked = {
            key for key, spec in ACTIVATION_SPECS.items()
            if spec.benchmarked
        }
        self.assertEqual(benchmarked, {"tanh", "relu", "sigmoid"})

    def test_unsupported_maxpooling_is_deferred(self) -> None:
        maxpooling = LAYER_SPECS["maxpooling"]
        self.assertEqual(maxpooling.status, "unchecked")
        self.assertEqual(maxpooling.priority, "defer")

    def test_support_matrix_is_serializable_shape(self) -> None:
        matrix = support_matrix()
        self.assertIn("layers", matrix)
        self.assertIn("activations", matrix)
        self.assertIn("benchmark_engines", matrix)
        self.assertEqual(matrix["benchmark_sizes"], BENCHMARK_SIZES)


if __name__ == "__main__":
    unittest.main()
