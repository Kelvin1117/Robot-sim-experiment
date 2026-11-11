import json
import os
from pathlib import Path
import socket
import subprocess
from multiprocessing.connection import Connection

import numpy as np
import torch


class Environment:
    def __init__(self,output,seed=0,settings=None):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        self.seed=seed
        self.settings=settings or dict(camera_resolution=128,warmup_steps=10,control_hz=20)
        py='/common/home/cs2202/miniconda3/envs/libero/bin/python'
        env=dict(os.environ,MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
                 OFFICIAL_OUTPUT=str(output),OFFICIAL_RUNTIME=str(output/'runtime'))
        prop=torch.cuda.get_device_properties(0)
        allocated=str(prop.uuid)
        if not allocated.startswith('GPU-'):allocated='GPU-'+allocated
        script='/common/home/cs2202/VLARLKit/experiments/github_official_ppo_benchmark/egl_devices.py'
        devices=json.loads(subprocess.check_output([py,script],env=env,text=True,timeout=30))
        matches=[d['egl_index'] for d in devices if d.get('uuid')==allocated]
        command_prefix=[]
        if len(matches)==1:
            env['MUJOCO_EGL_DEVICE_ID']=str(matches[0])
        else:
            # A100 is a compute GPU; use the existing tested Mesa/Xvfb path
            # instead of silently rendering on someone else's allocated GPU.
            env.update(MUJOCO_GL='glx',PYOPENGL_PLATFORM='glx',LIBGL_ALWAYS_SOFTWARE='1',__GLX_VENDOR_LIBRARY_NAME='mesa')
            command_prefix=['/usr/bin/xvfb-run','-a']
        left,right=socket.socketpair();self.pipe=Connection(left.detach())
        self.log=(output/'simulator.log').open('a')
        self.process=subprocess.Popen(command_prefix+[py,'-u','-B',str(Path(__file__).with_name('simulator_bridge.py')),str(right.fileno())],
            pass_fds=(right.fileno(),),env=env,stdout=self.log,stderr=subprocess.STDOUT)
        right.close()

    def request(self,message):
        self.pipe.send(message)
        if not self.pipe.poll(180):raise TimeoutError('LIBERO request timeout; see simulator.log')
        row=self.pipe.recv()
        if not row['ok']:raise RuntimeError(row['error'])
        return row

    def reset(self,task,initial):
        result=self.request(dict(command='reset',task=task,initial=initial,seed=self.seed,**self.settings))
        if result['control_hz']!=self.settings['control_hz']:raise ValueError('Simulator control frequency changed')
        return result['obs'],dict(instruction=result['instruction'])

    def step(self,action):
        result=self.request(dict(command='step',action=np.asarray(action).tolist()))
        return result['obs'],float(result['success']),result['success'],False,dict(is_success=result['success'],autoreset=False)

    def close(self):
        if self.process.poll() is None:
            try:self.request(dict(command='close'))
            except Exception:self.process.terminate()
        try:self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill();self.process.wait(timeout=10)
        self.pipe.close();self.log.close()
