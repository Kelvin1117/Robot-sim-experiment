import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproduce
from reproduction import container_entry as entry
from tools import download_assets

PREFIX = 'common/users/cs2202/test'


def tar_bytes(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w') as archive:
        for name, kind, value in entries:
            item = tarfile.TarInfo(name)
            item.type = kind
            if kind == tarfile.REGTYPE:
                item.size = len(value)
                archive.addfile(item, io.BytesIO(value))
            else:
                item.linkname = value
                archive.addfile(item)
    return output.getvalue()


class RestorationTests(unittest.TestCase):
    def test_files_hardlinks_and_relocated_alias(self):
        entries = [(PREFIX + '/data', tarfile.REGTYPE, b'weights'),
                   (PREFIX + '/hard', tarfile.LNKTYPE, PREFIX + '/data'),
                   (PREFIX + '/alias', tarfile.SYMTYPE, '/' + PREFIX + '/data'),
                   (PREFIX + '/python', tarfile.SYMTYPE, '/usr/bin/python3.11')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reproduce.extract_tar(io.BytesIO(tar_bytes(entries)), root)
            self.assertEqual((root / PREFIX / 'hard').stat().st_ino, (root / PREFIX / 'data').stat().st_ino)
            self.assertEqual((root / PREFIX / 'alias').resolve(), root / PREFIX / 'data')
            self.assertEqual((root / PREFIX / 'python').readlink(), Path('/usr/bin/python3.11'))

    def test_escape_paths_and_external_links_rejected(self):
        bad = [('common/users/cs2202/../escape', tarfile.REGTYPE, b'bad'),
               ('/tmp/escape', tarfile.REGTYPE, b'bad'),
               (PREFIX + '/alias', tarfile.SYMTYPE, '/tmp/outside'),
               (PREFIX + '/alias', tarfile.SYMTYPE, '../../../../../../outside')]
        for member in bad:
            with self.subTest(member=member), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    reproduce.extract_tar(io.BytesIO(tar_bytes([member])), Path(directory))

    def test_checksum_mismatch_and_incomplete_inventory_rejected(self):
        name = PREFIX + '/data'
        data = tar_bytes([(name, tarfile.REGTYPE, b'weights')])
        row = {'size': 7, 'type': '0', 'sha256': '0' * 64}
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'checksum'):
            reproduce.extract_tar(io.BytesIO(data), Path(directory), {name: row})
        row['sha256'] = hashlib.sha256(b'weights').hexdigest()
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'incomplete'):
            reproduce.extract_tar(io.BytesIO(data), Path(directory), {name: row, name + '-missing': row})

    @unittest.skipUnless(shutil.which('zstd'), 'zstd is required')
    def test_split_zstd_stream_restores_without_joined_archive(self):
        name = PREFIX + '/data'
        content = b'checkpoint' * 1000
        raw = tar_bytes([(name, tarfile.REGTYPE, content)])
        compressed = subprocess.check_output(['zstd', '-q', '-c'], input=raw)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts = []
            for i, offset in enumerate(range(0, len(compressed), 17)):
                path = root / ('part-' + str(i))
                path.write_bytes(compressed[offset:offset + 17])
                parts.append(path)
            dest = root / 'restore'
            dest.mkdir()
            count = reproduce.decompress(parts, dest, {name: {'size': len(content), 'type': '0', 'sha256': hashlib.sha256(content).hexdigest()}})
            self.assertEqual(count, 1)
            self.assertEqual((dest / name).read_bytes(), content)


class LaunchTests(unittest.TestCase):
    def args(self, root, **kwargs):
        defaults = dict(root=Path(root), action='evaluate', gpu='GPU-allocated', cpu_only=False,
                        run='result', from_run=None, resume=False, image='test-image')
        return argparse.Namespace(**{**defaults, **kwargs})

    def test_mounts_are_readonly_and_outputs_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'space in path'
            cmd = reproduce.docker_command(self.args(root))
            self.assertIn('device=GPU-allocated', cmd)
            self.assertIn(f'type=bind,src={root}/restored/common/users/cs2202,dst=/common/users/cs2202,readonly', cmd)
            self.assertIn(f'type=bind,src={root}/runs,dst=/outputs', cmd)
            self.assertFalse(root.exists())
            self.assertIn('none', cmd)

    def test_cpu_doctor_never_exposes_gpu(self):
        cmd = reproduce.docker_command(self.args('/tmp/work', action='doctor', cpu_only=True))
        self.assertNotIn('--gpus', cmd)
        self.assertIn('--cpu-only', cmd)

    def test_bad_run_and_gpu_selection_rejected(self):
        for kwargs in ({'run': '../original'}, {'gpu': 'all'}, {'gpu': '0,1'}, {'resume': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                reproduce.docker_command(self.args('/tmp/work', **kwargs))

    def test_dry_run_needs_no_docker_data_or_writable_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'missing'
            result = subprocess.run([sys.executable, str(reproduce.REPO / 'reproduce.py'), '--root', str(root),
                                     '--dry-run', 'evaluate'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)[0:2], ['docker', 'run'])
            self.assertFalse(root.exists())


class RunIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.frozen = root / 'frozen'
        self.outputs = root / 'outputs'
        self.frozen.mkdir()
        self.outputs.mkdir()
        (self.frozen / 'config.json').write_bytes(b'{\n  "episodes": 1200\n}\n')
        (self.frozen / 'best.pt').write_bytes(b'published-checkpoint')
        (self.frozen / 'source_manifest.json').write_text('{}')
        complete = dict(episodes=1200, config_sha256=entry.digest(self.frozen / 'config.json'),
                        best_sha256=entry.digest(self.frozen / 'best.pt'))
        (self.frozen / 'training_completed.json').write_text(json.dumps(complete))
        for key, value in [('FROZEN', self.frozen), ('OUTPUTS', self.outputs)]:
            patcher = mock.patch.object(entry, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_evaluation_copies_exact_settings_and_preserves_published_run(self):
        before = {p.name: p.read_bytes() for p in self.frozen.iterdir()}
        result = entry.prepare_run('evaluate', 'new-evaluation')
        for name in ('config.json', 'best.pt', 'training_completed.json'):
            self.assertEqual((result / name).read_bytes(), before[name])
        self.assertEqual({p.name: p.read_bytes() for p in self.frozen.iterdir()}, before)
        with self.assertRaises(FileExistsError):
            entry.prepare_run('evaluate', 'new-evaluation')

    def test_new_training_does_not_inherit_published_completion(self):
        result = entry.prepare_run('train', 'retrain')
        self.assertFalse((result / 'best.pt').exists())
        self.assertFalse((result / 'training_completed.json').exists())
        (result / 'latest.pt').write_bytes(b'resume-state')
        with self.assertRaisesRegex(RuntimeError, 'Training output exists'):
            entry.prepare_run('train', 'retrain')
        self.assertEqual(entry.prepare_run('train', 'retrain', resume=True), result)

    def test_tampered_selected_checkpoint_blocks_evaluation(self):
        (self.frozen / 'best.pt').write_bytes(b'modified')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            entry.prepare_run('evaluate', 'bad')
        self.assertFalse((self.outputs / 'bad').exists())

    def test_failed_smoke_prevents_training_exec(self):
        def smoke(*args, **kwargs):
            run = self.outputs / 'trial'
            (run / 'gpu_smoke.json').write_text(json.dumps({'passed': False}))
        with mock.patch.object(sys, 'argv', ['container_entry.py', 'train', '--run', 'trial']), \
                mock.patch.object(entry, 'environment', return_value={}), \
                mock.patch.object(entry.subprocess, 'run', side_effect=smoke), \
                mock.patch.object(entry.os, 'execve') as execute, \
                self.assertRaises(SystemExit):
            entry.main()
        execute.assert_not_called()


class PinnedDownloadTests(unittest.TestCase):
    def test_local_manifest_prevents_remote_manifest_substitution_and_supports_rerun(self):
        blob = b'backup-volume'
        url = 'https://github.com/Kelvin1117/Robot-sim-experiment/releases/download/retention-20261110/part'
        manifest = dict(uploaded=True, zstd_integrity_verified=True,
                        parts=[dict(file='part', size=len(blob), sha256=hashlib.sha256(blob).hexdigest(), url=url)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin = root / 'pinned.json'
            pin.write_text(json.dumps(manifest))
            argv = ['download_assets.py', '--directory', str(root / 'downloads'), '--manifest', str(pin)]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(download_assets.urllib.request, 'urlopen', return_value=io.BytesIO(blob)) as fetch:
                download_assets.main()
                fetch.assert_called_once_with(url, timeout=120)
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(download_assets.urllib.request, 'urlopen') as fetch:
                download_assets.main()
                fetch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
