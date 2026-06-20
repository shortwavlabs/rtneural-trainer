from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rttrainer.utils import write_json
from rttrainer.validation.parity import run_exported_json


class RTNeuralParityTests(unittest.TestCase):
    def test_keras_lstm_dense_json_runs_without_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.rtneural.json"
            write_json(
                model_path,
                {
                    "in_shape": [None, None, 1],
                    "layers": [
                        {
                            "type": "lstm",
                            "activation": "",
                            "shape": [None, None, 1],
                            "input_size": 1,
                            "hidden_size": 1,
                            "weights": [
                                [[0.0, 0.0, 1.0, 0.0]],
                                [[0.0, 0.0, 0.0, 0.0]],
                                [8.0, -8.0, 0.0, 8.0],
                            ],
                        },
                        {
                            "type": "dense",
                            "activation": "",
                            "shape": [None, None, 1],
                            "input_size": 1,
                            "output_size": 1,
                            "weights": [[[1.0]], [0.0]],
                        },
                    ],
                },
            )

            prediction = run_exported_json(model_path, [0.1, 0.2, -0.1])

        self.assertEqual(len(prediction), 3)
        self.assertGreater(prediction[0], 0.0)
        self.assertLess(prediction[2], prediction[1])

    def test_dense_only_activation_json_runs_without_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.rtneural.json"
            write_json(
                model_path,
                {
                    "in_shape": [None, None, 1],
                    "layers": [
                        {
                            "type": "dense",
                            "activation": "tanh",
                            "shape": [None, None, 2],
                            "weights": [[[1.0, -1.0]], [0.0, 0.0]],
                        },
                        {
                            "type": "activation",
                            "activation": "softmax",
                            "shape": [None, None, 2],
                            "weights": [],
                        },
                        {
                            "type": "dense",
                            "activation": "",
                            "shape": [None, None, 1],
                            "weights": [[[1.0], [-1.0]], [0.0]],
                        },
                    ],
                },
            )

            prediction = run_exported_json(model_path, [0.0, 0.5])

        self.assertAlmostEqual(prediction[0], 0.0, places=7)
        self.assertGreater(prediction[1], 0.0)

    def test_gru_dense_json_runs_without_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.rtneural.json"
            write_json(
                model_path,
                {
                    "in_shape": [None, None, 1],
                    "layers": [
                        {
                            "type": "gru",
                            "activation": "tanh",
                            "shape": [None, None, 1],
                            "input_size": 1,
                            "hidden_size": 1,
                            "weights": [
                                [[-8.0, -8.0, 1.0]],
                                [[0.0, 0.0, 0.0]],
                                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                            ],
                        },
                        {
                            "type": "dense",
                            "activation": "",
                            "shape": [None, None, 1],
                            "weights": [[[1.0]], [0.0]],
                        },
                    ],
                },
            )

            prediction = run_exported_json(model_path, [0.25, 0.25, -0.25])

        self.assertEqual(len(prediction), 3)
        self.assertGreater(prediction[0], 0.0)
        self.assertLess(prediction[2], prediction[1])

    def test_conv1d_batchnorm_prelu_json_runs_without_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.rtneural.json"
            write_json(
                model_path,
                {
                    "in_shape": [None, None, 1],
                    "layers": [
                        {
                            "type": "conv1d",
                            "activation": "",
                            "shape": [None, None, 1],
                            "kernel_size": [2],
                            "dilation": [1],
                            "groups": 1,
                            "weights": [[[[0.5]], [[1.0]]], [0.0]],
                        },
                        {
                            "type": "batchnorm",
                            "activation": "",
                            "shape": [None, None, 1],
                            "epsilon": 0.0,
                            "weights": [[2.0], [0.1], [0.5], [0.25]],
                        },
                        {
                            "type": "prelu",
                            "activation": "",
                            "shape": [None, None, 1],
                            "weights": [[[0.25]]],
                        },
                        {
                            "type": "dense",
                            "activation": "",
                            "shape": [None, None, 1],
                            "weights": [[[1.0]], [0.0]],
                        },
                    ],
                },
            )

            prediction = run_exported_json(model_path, [1.0, 2.0, -1.0])

        self.assertAlmostEqual(prediction[0], 2.1, places=7)
        self.assertAlmostEqual(prediction[1], 8.1, places=7)
        self.assertAlmostEqual(prediction[2], -0.475, places=7)


if __name__ == "__main__":
    unittest.main()
