import json,math
from pathlib import Path
import h5py
p=Path(__file__).resolve().parents[1]
f=p/'configs/full.json';c=json.loads(f.read_text());total=0
for path in sorted(Path(c['data']['directory']).glob('*_demo.hdf5')):
 with h5py.File(path,'r') as h:
  keys=sorted(h['data'],key=lambda k:int(k.split('_')[-1]))[:-c['data']['validation_episodes']]
  total+=sum(len(h['data'][k]['actions'])-1 for k in keys)
s=c['stage1'];unit=s['validation_every']
for key,epochs in [('initial_steps',s['requested_initial_epochs']),('steps',s['requested_maximum_epochs'])]:
 s[key]=math.ceil(total*epochs/s['batch_size']/unit)*unit
s['min_completed_steps']=s['initial_steps'];s['training_split_observations']=total
f.write_text(json.dumps(c,indent=2)+'\n')
print(json.dumps(s,indent=2))
