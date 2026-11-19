"""Retain smoke model deltas/evidence, discard only smoke optimizer retention.

Never accepts a formal run. Original SFT and other experiments are untouched.
"""
import argparse
import json
from pathlib import Path
import os

from storage import digest,load,save,atomic_json

p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
run=args.run.resolve();port=Path(__file__).resolve().parents[1]
if not run.is_relative_to(port/'runs') or not run.name.startswith('smoke-'):
    raise ValueError('Only this project smoke runs may be compacted')
config=json.loads((run/'config.json').read_text())
if config['stage1']['steps']>2:raise ValueError('Not a smoke budget')
if not (run/'shell_exit.txt').exists():raise ValueError('Smoke job must have exited')
report=[];inodes={}
for name in ['best','latest']:
    path=run/'stage1'/(name+'.pt')
    if not path.exists():continue
    inode=(path.stat().st_dev,path.stat().st_ino)
    archived=path.with_name(name+'-inference-delta.pt')
    if inode in inodes:
        original_hash,previous,archived_hash=inodes[inode]
        os.link(previous,archived)
    else:
        state=load(path)
        save(archived,{k:state[k] for k in ['format_version','stage','step','config_hash','base_sha256','delta']})
        del state
        original_hash=digest(path);archived_hash=digest(archived)
        inodes[inode]=(original_hash,archived,archived_hash)
    report.append(dict(original=str(path),original_sha256=original_hash,archived=str(archived),archived_sha256=archived_hash,
                       optimizer_retention='removed for finished smoke only; inference delta retained; not a resume checkpoint'))
    path.unlink()
atomic_json(run/'smoke_retention.json',report)
