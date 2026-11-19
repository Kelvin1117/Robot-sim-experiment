import pathlib,sys,json,hashlib,numpy as np
from simulator import Environment
out=pathlib.Path(sys.argv[1]);report=[]
for warm in [10]:
 e=Environment(out/('warm'+str(warm)),settings=dict(camera_resolution=128,warmup_steps=warm,control_hz=20));prior={}
 try:
  for i,t in enumerate([0,3,0,3,3,3]):
   e.seed=910100024 if t==0 else 910100324
   raw,_=e.reset(t,24)
   row=dict(warmup=warm,iteration=i,task=t,hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in raw.items()})
   if t in prior:row['differences']={k:dict(max_abs=float(np.abs(v.astype(float)-prior[t][k].astype(float)).max()),changed=int(np.count_nonzero(v!=prior[t][k]))) for k,v in raw.items()}
   prior[t]={k:v.copy() for k,v in raw.items()};report.append(row);print(json.dumps(row),flush=True)
   for step in range(20): e.step(np.random.RandomState(1000*t+step).uniform(-.1,.1,7))
 finally:e.close()
(out/'diagnostic.json').write_text(json.dumps(report,indent=2))
