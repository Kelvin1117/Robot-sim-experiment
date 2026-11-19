import argparse
import json
from pathlib import Path
import numpy as np
import torch
from run import build,amp
from simulator import Environment
from storage import atomic_json

p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run',type=Path,required=True)
args=p.parse_args();config=json.loads(Path(args.config).read_text())
adapter,pre,data=build(config);adapter.eval();env=Environment(args.run/'live_interface',settings=config['environment'])
report={}
try:
    raw,info=env.reset(3,22)
    assert info['instruction']==data.instructions[3]
    episode=data.episode_ids(3)[0];g=data.files[3]['data'][episode]
    target=g['obs/ee_states'][0,:3]
    errors=[]
    for index in [0,1]:
        result=env.request(dict(command='dataset_state',state=g['states'][index].tolist()))
        errors.append(float(np.linalg.norm(result['obs']['robot0_eef_pos']-target)))
    report['post_action_alignment_errors']=errors
    if not errors[1]<errors[0] or errors[1]>0.01:
        raise ValueError(f'Post-action alignment not verified: {errors}')
    # Dataset normalization agrees with the native OpenPI implementation.
    from openpi.transforms import Normalize,Unnormalize,NormStats
    stats={'actions':NormStats(**pre.stats['actions'])}
    physical=g['actions'][1:11].astype(np.float32)
    normalized=Normalize(stats,use_quantiles=True)({'actions':physical})['actions']
    np.testing.assert_allclose(normalized,pre.normalize(physical,'actions'),atol=1e-6)
    restored=pre.denormalize(torch.tensor(normalized)).numpy()
    np.testing.assert_allclose(restored,physical,atol=1e-6)
    native=Unnormalize(stats,use_quantiles=True)({'actions':normalized})['actions']
    np.testing.assert_allclose(restored,native,atol=1e-6)
    raw,info=env.reset(3,22);batch=pre.live(raw,info['instruction'])
    with torch.no_grad(),amp():
        z=adapter.encode(batch,False);base=adapter.base_chunk(batch,z)
    assert base.shape==(1,config['model']['horizon'],7) and torch.isfinite(base).all()
    # Compare actual environment commands under identical true simulator state.
    actual=env.request(dict(command='dataset_state',state=g['states'][20].tolist()))
    baseline=np.zeros(7,np.float32);baseline[-1]=-1
    before,_,_,_,_=env.step(baseline)
    perturbed=baseline.copy();perturbed[0]=0.1
    env.request(dict(command='dataset_state',state=g['states'][20].tolist()))
    changed,_,_,_,_=env.step(perturbed)
    assert np.isfinite(changed['robot0_eef_pos']).all()
    movement=float(np.linalg.norm(changed['robot0_eef_pos']-before['robot0_eef_pos']))
    assert movement>1e-8
    report.update(status='PASS',state_shape=list(batch['state'].shape),base_shape=list(base.shape),
        normalization_native_parity=True,action_offset=1,control_hz=20,gpu=torch.cuda.get_device_name(),
        changed_command_eef_difference=movement,
        pretrained_load_audit=adapter.core.pretrained_load_audit,
        gripper_negative='open: simulator warm-up uses -1',
        note='live adapter inference and physical command test; not a success-rate baseline')
    atomic_json(args.run/'live_interface.json',report)
finally:env.close();data.close()
