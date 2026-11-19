import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('audit', Path(__file__).parents[1] / 'scripts/audit_checkpoint.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class HeaderTests(unittest.TestCase):
    def check_header(self, descriptor, payload, expected_error=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.safetensors'
            encoded = json.dumps(descriptor).encode()
            path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
            if expected_error:
                with self.assertRaises(ValueError):
                    audit.read_header(path)
            else:
                return audit.read_header(path)

    def test_counts_stored_values_without_counting_tied_alias_metadata(self):
        tensors, counts = self.check_header({'w': {'dtype': 'F32', 'shape': [2, 3],
            'data_offsets': [0, 24]}, '__metadata__': {'alias': 'w'}}, bytes(24))
        self.assertEqual(counts, {'F32': 6})
        self.assertEqual(len(tensors), 1)

    def test_truncated_payload_rejected(self):
        self.check_header({'w': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}}, bytes(4), True)

    def test_overlapping_tensors_rejected(self):
        desc = {k: {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]} for k in ['a', 'b']}
        self.check_header(desc, bytes(4), True)

    def test_declared_shape_cannot_hide_missing_values(self):
        self.check_header({'w': {'dtype': 'F32', 'shape': [8], 'data_offsets': [0, 4]}}, bytes(4), True)

    def test_reversed_quantiles_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stats.json'
            stats = {key: {'mean': [0]*dim, 'std': [1]*dim, 'q01': [1]*dim, 'q99': [-1]*dim}
                     for key, dim in [('state', 8), ('actions', 7)]}
            path.write_text(json.dumps({'norm_stats': stats}))
            with self.assertRaises(ValueError):
                audit.check_stats(path)


if __name__ == '__main__':
    unittest.main()
