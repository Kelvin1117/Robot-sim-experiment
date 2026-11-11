from pathlib import Path
import json
work=Path('/common/users/cs2202/robot-experiment-assets')
rows=[json.loads(s) for s in (work/'archive_files.jsonl').read_text().splitlines()]
index={r['path']:r for r in rows}
root=Path('/common/users/cs2202/dygro_pi05_phase012_20261106')
run=root/'oneshot_retention/runs/retention-20261109T224843Z'
parent=root/'oneshot_paper_v2/runs/paper-v2-20261108T174348Z'
checks=[]
def digest(path):
 name=str(path).lstrip('/')
 row=index[name]
 visited=set()
 while row['type']=='1':
  if row['path'] in visited:raise RuntimeError('Cyclic hardlink')
  visited.add(row['path']);row=index[row['linkname']]
 return row.get('sha256')
def check(path,expected,label):
 actual=digest(path)
 checks.append({'path':str(path),'check':label,'expected_sha256':expected,'archived_sha256':actual,'matches':actual==expected})
for name,expected in json.loads((run/'source_manifest.json').read_text()).items():check(run/name,expected,'frozen run source manifest')
completed=json.loads((run/'training_completed.json').read_text())
check(run/'best.pt',completed['best_sha256'],'selected recovery checkpoint')
for name,expected in completed['source_hashes'].items():check(parent/name,expected,'frozen parent source hashes')
config=json.loads((parent/'config.json').read_text())['model']
for key in ('weights','normalization','tokenizer'):check(Path(config[key]),config[key+'_sha256'],'parent model configuration')
for row in json.loads((root/'oneshot_paper_v2/reports/external_dependency_hashes.json').read_text()):check(Path(row['path']),row['sha256'],'external dependency hash')
readmes=[r['path'] for r in rows if Path(r['path']).name.lower().startswith('readme')]
datasets=[r['path'] for r in rows if r['path'].startswith(str(root/'oneshot_openpi/data/libero_goal').lstrip('/')+'/') and r['path'].endswith('.hdf5')]
result={'hash_checks':checks,'all_hash_checks_match':all(r['matches'] for r in checks),'readme_entries':readmes,'libero_goal_hdf5_files':len(datasets),'archive_entries':len(rows),'passed':all(r['matches'] for r in checks) and not readmes and len(datasets)==10}
(work/'integrity_report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'hash_checks':len(checks),'dataset_files':len(datasets),'readme_entries':len(readmes),'archive_entries':len(rows)}))
if not result['passed']:raise SystemExit(1)
