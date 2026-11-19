"""Official LIBERO HDF5: post-action obs[i] conditions actions[i+1:]."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch


TASKS = ['open_the_middle_drawer_of_the_cabinet','put_the_bowl_on_the_stove',
         'put_the_wine_bottle_on_top_of_the_cabinet','open_the_top_drawer_and_put_the_bowl_inside',
         'put_the_bowl_on_top_of_the_cabinet','push_the_plate_to_the_front_of_the_stove',
         'put_the_cream_cheese_in_the_bowl','turn_on_the_stove','put_the_bowl_on_the_plate',
         'put_the_wine_bottle_on_the_rack']


def validate_data(directory):
    report=[]
    for task,name in enumerate(TASKS):
        path=Path(directory)/(name+'_demo.hdf5')
        with h5py.File(path,'r') as f:
            meta=f['data'].attrs
            env=json.loads(meta['env_args'])['env_kwargs']
            assert env['control_freq']==20
            assert env['controller_configs']['type']=='OSC_POSE'
            instruction=json.loads(meta['problem_info'])['language_instruction']
            assert instruction.replace(' ','_')==name
            for key,g in f['data'].items():
                n=len(g['actions'])
                assert n>2 and g['actions'].shape==(n,7)
                for k in ['agentview_rgb','eye_in_hand_rgb','ee_states','gripper_states']:
                    assert len(g['obs'][k])==n
                assert g['obs/ee_states'].shape==(n,6) and g['obs/gripper_states'].shape==(n,2)
                assert np.isfinite(g['actions'][:]).all() and np.isfinite(g['obs/ee_states'][:]).all()
                assert np.isfinite(g['obs/gripper_states'][:]).all()
                assert np.max(np.abs(g['actions'][:]))<1.01
                assert len(g['dones'])==n and len(g['rewards'])==n
                assert not g['dones'][:-1].any() and g['dones'][-1]==1
                assert np.isfinite(g['rewards'][:]).all()
                assert g['states'].shape[0]==n
                # MuJoCo flattened state starts with simulation time.
                np.testing.assert_allclose(np.diff(g['states'][:,0]),1/20,atol=1e-6)
            report.append(dict(task=task,instruction=instruction,path=str(path.resolve()),
                               episodes=len(f['data']),frames=sum(len(g['actions']) for g in f['data'].values()),
                               action_offset=1,control_hz=20,timestamp_source='states[:,0]',
                               reward_source='stored LIBERO demonstration annotation',
                               observation_timing='post-action, verified against create_dataset.py; replay smoke required'))
    return report


class GoalData:
    def __init__(self,directory,preprocessor,horizon,validation_episodes):
        self.directory,self.pre,self.horizon=Path(directory),preprocessor,horizon
        self.validation_episodes=validation_episodes
        self.files=[h5py.File(self.directory/(name+'_demo.hdf5'),'r') for name in TASKS]
        self.instructions=[json.loads(f['data'].attrs['problem_info'])['language_instruction'] for f in self.files]

    def training_frames(self):
        return sum(len(self.files[t]['data'][ep]['actions'])-1
                   for t in range(10) for ep in self.episode_ids(t))

    def close(self):
        for f in self.files:f.close()

    def episode_ids(self,task,validation=False):
        keys=sorted(self.files[task]['data'],key=lambda x:int(x.split('_')[-1]))
        return keys[-self.validation_episodes:] if validation else keys[:-self.validation_episodes]

    def frame(self,task,episode,index):
        g=self.files[task]['data'][episode]['obs']
        state=np.concatenate([g['ee_states'][index],g['gripper_states'][index]])
        return self.pre.frame([g['agentview_rgb'][index],g['eye_in_hand_rgb'][index]],state,self.instructions[task])

    def chunk(self,task,episode,start):
        g=self.files[task]['data'][episode]
        begin=start+1
        end=min(begin+self.horizon,len(g['actions']))
        if not 0<=start<begin<end<=len(g['actions']):raise ValueError('No next action or crossed episode')
        duration=end-begin
        actions=np.zeros((self.horizon,7),np.float32)
        actions[:duration]=self.pre.normalize(g['actions'][begin:end],'actions')
        row=self.frame(task,episode,start)
        row['action']=torch.tensor(actions)
        return row,torch.arange(self.horizon)<duration,begin,end

    def sample(self,batch_size,generator,validation=False):
        rows,masks=[],[]
        for _ in range(batch_size):
            task=int(generator.integers(10))
            keys=self.episode_ids(task,validation)
            episode=keys[int(generator.integers(len(keys)))]
            start=int(generator.integers(len(self.files[task]['data'][episode]['actions'])-1))
            row,mask,_,_=self.chunk(task,episode,start)
            rows.append(row);masks.append(mask)
        return self.pre.batch(rows),torch.stack(masks).to(self.pre.device)

    @torch.no_grad()
    def populate(self,replay,adapter,gamma,episodes_per_task,on_row=None,max_chunks_per_episode=0):
        for task in range(10):
            for episode in self.episode_ids(task)[:episodes_per_task]:
                g=self.files[task]['data'][episode]
                rewards=g['rewards'][:].astype(np.float32)
                mc=np.zeros_like(rewards)
                value=0.
                for i in reversed(range(len(rewards))):
                    value=float(rewards[i])+gamma*(not bool(g['dones'][i]))*value
                    mc[i]=value
                for start in range(0,len(rewards)-1,self.horizon):
                    if max_chunks_per_episode and start//self.horizon>=max_chunks_per_episode:break
                    row,mask,begin,end=self.chunk(task,episode,start)
                    batch=self.pre.batch([row]);following=self.pre.batch([self.frame(task,episode,end-1)])
                    z=adapter.encode(batch,False);zn=adapter.encode(following,False)
                    replay.add(dict(latent=z[0].float(),base=adapter.base_chunk(batch,z)[0],action=batch['action'][0],
                        next_latent=zn[0].float(),next_base=adapter.base_chunk(following,zn)[0],
                        reward=float(sum(gamma**i*r for i,r in enumerate(rewards[begin:end]))),
                        duration=end-begin,terminated=bool(g['dones'][end-1]),truncated=False,
                        task_id=task,valid_mask=mask,mc_return=float(mc[begin])))
                    if on_row:on_row(task,episode,len(replay))
