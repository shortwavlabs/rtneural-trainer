from __future__ import annotations

import unittest

from rttrainer.training.device import normalize_device_preference


class TrainingDeviceTests(unittest.TestCase):
    def test_normalizes_device_preferences(self) -> None:
        self.assertEqual(normalize_device_preference(None), "auto")
        self.assertEqual(normalize_device_preference(""), "auto")
        self.assertEqual(normalize_device_preference("CPU"), "cpu")
        self.assertEqual(normalize_device_preference("metal"), "mps")
        self.assertEqual(normalize_device_preference("gpu"), "cuda")
        self.assertEqual(normalize_device_preference("cuda:0"), "cuda")
        self.assertEqual(
            normalize_device_preference("tensorflow-cpu:/physical_device:CPU:0"),
            "cpu",
        )
        self.assertEqual(
            normalize_device_preference("tensorflow-gpu:/physical_device:GPU:0"),
            "auto",
        )
        self.assertEqual(
            normalize_device_preference("tensorflow-metal:/physical_device:GPU:0"),
            "mps",
        )

    def test_rejects_unknown_device_preference(self) -> None:
        with self.assertRaises(RuntimeError):
            normalize_device_preference("neural-engine")


if __name__ == "__main__":
    unittest.main()
