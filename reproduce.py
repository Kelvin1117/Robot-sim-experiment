#!/usr/bin/env python3
"""重現 2026-11-10 retention 實驗（Linux x86_64）。不需手改原機路徑。

先安裝 Docker、NVIDIA Container Toolkit、zstd；預留約 110 GiB 磁碟。
原實驗使用 A100 40 GB、96 GiB RAM；其他 GPU 尚未驗證。

  python3 reproduce.py --root /data/robot setup
  python3 reproduce.py --root /data/robot --gpu 0 doctor
  python3 reproduce.py --root /data/robot --gpu 0 evaluate
  python3 reproduce.py --root /data/robot --gpu 0 --run retrain train
  python3 reproduce.py --root /data/robot --gpu 0 --from-run retrain evaluate

setup 下載並校驗 47 卷、串流解壓，再建置容器；可重跑以略過已校驗的分卷。
原始快照以唯讀方式掛載；輸出在 ROOT/runs，快取在 ROOT/cache。
train 會先執行原實驗的 GPU smoke，再從 frozen parent 訓練 1200 episodes。
evaluate 預設重評發布的 best.pt：parent/recovered 各 600 episodes。
Slurm：sbatch reproduction/slurm.sh --root /data/robot --run retrain train
目前沒有在全新 GPU 容器中完成端到端驗證；doctor/smoke 用於部署後驗證。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
import threading
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parent
IMAGE = 'robot-sim-experiment:20261110'
PREFIXES = ('common/home/cs2202', 'common/users/cs2202')


def command(args):
    print('+ ' + ' '.join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True)


def archive_path(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not any(name == root or name.startswith(root + '/') for root in PREFIXES):
        raise ValueError('Unexpected archive path: ' + name)
    return p


def extract_tar(stream, destination, expected=None):
    """Extract regular files/links only; never write through an escaping parent."""
    destination = destination.resolve()
    seen = set()
    with tarfile.open(fileobj=stream, mode='r|') as archive:
        for member in archive:
            rel = archive_path(member.name)
            target = destination / str(rel)
            if not target.parent.resolve().is_relative_to(destination):
                raise ValueError('Archive parent escapes destination: ' + member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            row = expected[member.name] if expected is not None else None
            if row and (row['size'] != member.size or row['type'] != member.type.decode('ascii')):
                raise ValueError('Archive inventory differs: ' + member.name)
            if member.name in seen:
                raise ValueError('Duplicate archive entry: ' + member.name)
            seen.add(member.name)
            if member.isdir():
                if target.is_symlink():
                    raise ValueError('Directory is a symlink: ' + member.name)
                target.mkdir(exist_ok=True)
            elif member.isfile():
                digest = hashlib.sha256()
                with archive.extractfile(member) as source, target.open('xb') as output:
                    while block := source.read(4 * 1024 * 1024):
                        output.write(block)
                        digest.update(block)
                if row and digest.hexdigest() != row['sha256']:
                    raise ValueError('Extracted file checksum differs: ' + member.name)
                target.chmod(member.mode & 0o777)
            elif member.islnk():
                source = destination / str(archive_path(member.linkname))
                if not source.resolve().is_relative_to(destination) or source.is_symlink():
                    raise ValueError('Invalid hardlink: ' + member.name)
                os.link(source, target)
            elif member.issym():
                link = member.linkname
                if link.startswith('/common/'):
                    linked = archive_path(link.lstrip('/'))
                    link = posixpath.relpath(str(linked), str(rel.parent))
                elif link.startswith('/'):
                    if link != '/usr/bin/python3.11':
                        raise ValueError('Unexpected external symlink: ' + member.name)
                else:
                    linked = posixpath.normpath(posixpath.join(str(rel.parent), link))
                    archive_path(linked)
                target.symlink_to(link)
            else:
                raise ValueError('Unsupported archive entry: ' + member.name)
            if len(seen) % 10000 == 0:
                print('Restored', len(seen), 'paths', flush=True)
    if expected is not None and seen != set(expected):
        raise ValueError('Archive inventory is incomplete')
    return len(seen)


def decompress(paths, destination, expected=None):
    process = subprocess.Popen(['zstd', '-dc'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    errors = []

    def feed():
        try:
            for path in paths:
                with path.open('rb') as source:
                    shutil.copyfileobj(source, process.stdin, 4 * 1024 * 1024)
            process.stdin.close()
        except (OSError, ValueError) as exc:
            errors.append(exc)

    writer = threading.Thread(target=feed, daemon=True)
    writer.start()
    try:
        count = extract_tar(process.stdout, destination, expected)
        while process.stdout.read(1024 * 1024):
            pass
        writer.join()
        stderr = process.stderr.read().decode(errors='replace')
        if process.wait() or errors:
            raise RuntimeError('Archive decompression failed: ' + stderr)
        return count
    except BaseException:
        process.kill()
        process.wait()
        writer.join(timeout=5)
        raise
    finally:
        process.stdout.close()
        process.stderr.close()


def build(args):
    command(['docker', 'build', '--platform', 'linux/amd64', '-t', args.image,
             '-f', REPO / 'reproduction/Dockerfile', REPO])


def setup(args):
    if not shutil.which('zstd'):
        raise RuntimeError('Install zstd before setup (Ubuntu: sudo apt-get install zstd).')
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (REPO / 'assets/asset_manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
    restored = root / 'restored'
    marker = restored / '.complete.json'
    if marker.exists():
        if json.loads(marker.read_text())['manifest_sha256'] != fingerprint:
            raise RuntimeError('Existing restored data belongs to a different manifest; choose another --root.')
        print('Existing verified restoration found.', flush=True)
    else:
        partial = root / 'restored.partial'
        if partial.exists() or restored.exists():
            raise RuntimeError('Incomplete restoration exists; inspect/remove restored.partial or choose another --root.')
        archives = root / 'archives'
        remaining = sum(max(0, p['size'] - ((archives / p['file']).stat().st_size if (archives / p['file']).exists() else 0)) for p in manifest['parts'])
        needed = manifest['logical_regular_bytes'] + remaining + 10 * 1024**3
        if shutil.disk_usage(root).free < needed:
            raise RuntimeError(f'Need approximately {needed / 1024**3:.1f} GiB free at {root}; user quotas may be lower.')
        command([sys.executable, REPO / 'tools/download_assets.py', '--directory', archives,
                 '--manifest', REPO / 'assets/asset_manifest.json'])
        inventory = {r['path']: r for r in (json.loads(line) for line in (archives / 'archive_files.jsonl').read_text().splitlines())}
        partial.mkdir()
        count = decompress([archives / p['file'] for p in manifest['parts']], partial, inventory)
        for item in manifest['supplemental_archives']:
            decompress([archives / item['file']], partial)
        (partial / '.complete.json').write_text(json.dumps({'manifest_sha256': fingerprint, 'entries': count}, indent=2) + '\n')
        partial.rename(restored)
    if not args.skip_build:
        build(args)
    print('Ready. Run doctor, then evaluate or train. See python3 reproduce.py --help.', flush=True)


def docker_command(args):
    root = args.root
    if args.resume and args.action != 'train':
        raise ValueError('--resume is only supported for train')
    if args.from_run and args.action != 'evaluate':
        raise ValueError('--from-run is only supported for evaluate')
    if args.cpu_only and args.action != 'doctor':
        raise ValueError('--cpu-only is only supported for doctor')
    gpu = args.gpu or os.environ.get('SLURM_JOB_GPUS') or os.environ.get('CUDA_VISIBLE_DEVICES') or '0'
    if not args.cpu_only and not re.fullmatch(r'(?:\d+|GPU-[A-Za-z0-9-]+)', gpu):
        raise ValueError('Select one allocated GPU index or UUID using --gpu.')
    name = args.run or (args.action + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    for value in (name, args.from_run):
        if value is not None and (value in ('.', '..') or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', value)):
            raise ValueError('Run names must be simple directory names, not paths.')
    mounts = [(root / 'restored/common/home/cs2202', '/common/home/cs2202', True),
              (root / 'restored/common/users/cs2202', '/common/users/cs2202', True),
              (REPO, '/reproduction', True), (root / 'runs', '/outputs', False),
              (root / 'cache', '/cache', False)]
    cmd = ['docker', 'run', '--rm', '--init', '--platform', 'linux/amd64', '--network', 'none',
           '--user', f'{os.getuid()}:{os.getgid()}', '--shm-size', '8g', '-e', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics']
    if not args.cpu_only:
        cmd += ['--gpus', 'device=' + gpu]
    for source, dest, readonly in mounts:
        if ',' in str(source):
            raise ValueError('Docker mount paths cannot contain commas.')
        cmd += ['--mount', f'type=bind,src={source},dst={dest}' + (',readonly' if readonly else '')]
    cmd += [args.image, args.action, '--run', name]
    if args.from_run:
        cmd += ['--from-run', args.from_run]
    if args.resume:
        cmd += ['--resume']
    if args.cpu_only:
        cmd += ['--cpu-only']
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=REPO / '.robot-experiment')
    parser.add_argument('--image', default=IMAGE)
    parser.add_argument('--gpu', help='Single allocated GPU index or UUID; Slurm allocation is used when available')
    parser.add_argument('--run', help='Output run name; train uses this same name for smoke and training')
    parser.add_argument('--from-run', help='Evaluate a completed retraining run instead of published best.pt')
    parser.add_argument('--resume', action='store_true', help='Resume an existing training run')
    parser.add_argument('--cpu-only', action='store_true', help='Check files and imports without GPU (doctor only)')
    parser.add_argument('--dry-run', action='store_true', help='Print the Docker launch as JSON without creating files')
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('setup').add_argument('--skip-build', action='store_true', help='Download/extract only; build later')
    for action in ('build', 'doctor', 'smoke', 'train', 'evaluate'):
        sub.add_parser(action)
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    try:
        if args.action == 'setup':
            if args.dry_run:
                raise ValueError('--dry-run is supported for doctor/smoke/train/evaluate')
            setup(args)
        elif args.action == 'build':
            if args.dry_run:
                raise ValueError('--dry-run is supported for doctor/smoke/train/evaluate')
            build(args)
        else:
            cmd = docker_command(args)
            if args.dry_run:
                print(json.dumps(cmd, ensure_ascii=False, indent=2))
                return
            if not (args.root / 'restored/.complete.json').is_file():
                raise RuntimeError('Run setup first using the same --root.')
            for name in ('runs', 'cache'):
                (args.root / name).mkdir(parents=True, exist_ok=True)
            command(cmd)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
