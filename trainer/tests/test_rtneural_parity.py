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


if __name__ == "__main__":
    unittest.main()
