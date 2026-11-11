"""Pinned, direct-to-storage downloads; no second HF cache copy."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def fetch(item):
    url, path, size, expected_hash = item
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == size:
        return {'path': str(path), 'status': 'existing', 'bytes': size}
    partial = path.with_name(path.name + '.download')
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as stream:
        while block := response.read(8 * 1024 * 1024):
            stream.write(block)
            digest.update(block)
    if partial.stat().st_size != size or (expected_hash and digest.hexdigest() != expected_hash):
        raise ValueError(f'Download integrity failure: {path}')
    partial.replace(path)
    row = dict(path=str(path), bytes=size, sha256=digest.hexdigest(), status='downloaded')
    print(json.dumps(row), flush=True)
    return row


def main():
    model = json.loads((ROOT/'reports/hf_current.json').read_text())
    items = []
    for entry in model['siblings']:
        if not entry['rfilename'].startswith('5000/'):
            continue
        relative = entry['rfilename']
        items.append((f"https://huggingface.co/{model['id']}/resolve/{model['sha']}/{relative}",
                      '/common/users/cs2202/models/pi05_libero_one_shot/'+relative,
                      entry['size'], entry.get('lfs', {}).get('sha256')))
    with urllib.request.urlopen('https://huggingface.co/api/datasets/yifengzhu-hf/LIBERO-datasets', timeout=30) as response:
        dataset = json.load(response)
    revision = dataset['sha']
    (ROOT/'reports/official_dataset_revision.json').write_text(json.dumps({'revision':revision}))
    for entry in json.loads((ROOT/'reports/official_dataset_tree.json').read_text()):
        relative = entry['path']
        if relative.endswith('.hdf5'):
            items.append((f'https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/{revision}/{relative}',
                          str(ROOT/'data'/relative), entry['size'], entry.get('lfs', {}).get('oid')))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(fetch, items))
    (ROOT/'reports/asset_downloads.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
