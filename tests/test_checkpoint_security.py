"""Security regression checks; no simulator, GPU or training is started."""
import importlib.util
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]

class NonTensorPayload:
    def __reduce__(self):
        return (eval, ("1 + 1",))

@unittest.skipIf(torch is None, "PyTorch is required for checkpoint security checks")
class CheckpointSecurityTests(unittest.TestCase):
    def modules(self):
        for path in sorted(ROOT.rglob("storage.py")):
            spec = importlib.util.spec_from_file_location("checked_storage", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            yield path, module

    def test_public_best_loads_safely(self):
        best = ROOT / "experiments/dygro-20261117-985/best.pt"
        with mock.patch.dict(os.environ, {}, clear=True):
            for path, module in self.modules():
                with self.subTest(path=str(path)):
                    state = module.load(best)
                    self.assertEqual(state["episode"], 600)
                    self.assertEqual(len({k.split(".")[1] for k in state["correction"] if k.startswith("experts.")}), 4)

    def test_non_tensor_pickle_rejected_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "untrusted.pt"
            torch.save(NonTensorPayload(), path)
            with mock.patch.dict(os.environ, {}, clear=True):
                for source, module in self.modules():
                    with self.subTest(path=str(source)), self.assertRaises(pickle.UnpicklingError):
                        module.load(path)

    def test_trusted_mode_requires_explicit_opt_in(self):
        with mock.patch.dict(os.environ, {"DYGRO_TRUST_CHECKPOINT_PICKLE": "1"}, clear=True):
            for path, module in self.modules():
                with self.subTest(path=str(path)), mock.patch.object(module.torch, "load", return_value={}) as load:
                    module.load("trusted-local-resume.pt")
                    self.assertFalse(load.call_args.kwargs["weights_only"])

if __name__ == "__main__":
    unittest.main()
