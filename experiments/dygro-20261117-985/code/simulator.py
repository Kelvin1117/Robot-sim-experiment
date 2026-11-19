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
        py='/common/home/researcher/miniconda3/envs/libero/bin/python'
        env=dict(os.environ,MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
                 OFFICIAL_OUTPUT=str(output),OFFICIAL_RUNTIME=str(output/'runtime'))
        prop=torch.cuda.get_device_properties(0)
        allocated=str(prop.uuid)
        if not allocated.startswith('GPU-'):allocated='GPU-'+allocated
        script=str(Path(__file__).with_name('egl_devices.py'))
        devices=json.loads(subprocess.check_output([py,script],env=env,text=True,timeout=30))
        matches=[d['egl_index'] for d in devices if d.get('uuid')==allocated]
        if len(matches)!=1:
            raise RuntimeError('Allocated GPU has no unique EGL renderer; refusing CPU fallback')
        env['MUJOCO_EGL_DEVICE_ID']=str(matches[0])
        for key in ['LIBGL_ALWAYS_SOFTWARE','__GLX_VENDOR_LIBRARY_NAME']:
            env.pop(key, None)
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   BLIS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
                   NUMBA_NUM_THREADS='1', PYTHONHASHSEED='0')
        (output/'renderer.json').write_text(json.dumps(dict(backend='egl',gpu_uuid=allocated,egl_index=matches[0]),indent=2))
        # Native MuJoCo / OpenGL state must not cross episode boundaries.
        # Reconstructing only the Python environment still shares process globals,
        # delayed context destruction, and renderer caches with prior rollouts.
        self._python = py
        self._worker_env = env
        self._output = output
        self.process = None
        self.pipe = None
        self.log = None
        self._closed = False

    def _start_worker(self):
        left, right = socket.socketpair()
        self.pipe = Connection(left.detach())
        self.log = (self._output/'simulator.log').open('a')
        try:
            self.process = subprocess.Popen(
                [self._python, '-u', '-B',
                 str(Path(__file__).with_name('simulator_bridge.py')), str(right.fileno())],
                pass_fds=(right.fileno(),), env=self._worker_env,
                stdout=self.log, stderr=subprocess.STDOUT)
        except BaseException:
            self.pipe.close()
            self.log.close()
            self.pipe = self.log = None
            raise
        finally:
            right.close()

    def _stop_worker(self):
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    self.request(dict(command='close'))
                except Exception:
                    process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        finally:
            self.pipe.close()
            self.log.close()
            self.process = self.pipe = self.log = None

    def request(self,message):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError('Simulator worker is not running; reset the episode first')
        self.pipe.send(message)
        if not self.pipe.poll(180):raise TimeoutError('LIBERO request timeout; see simulator.log')
        row=self.pipe.recv()
        if not row['ok']:raise RuntimeError(row['error'])
        return row

    def reset(self,task,initial):
        if self._closed:
            raise RuntimeError('Cannot reset a closed simulator')
        self._stop_worker()
        self._start_worker()
        result=self.request(dict(command='reset',task=task,initial=initial,seed=self.seed,**self.settings))
        if result['control_hz']!=self.settings['control_hz']:raise ValueError('Simulator control frequency changed')
        return result['obs'],dict(instruction=result['instruction'])

    def step(self,action):
        result=self.request(dict(command='step',action=np.asarray(action).tolist()))
        return result['obs'],1.0 if result['success'] else -0.01,result['success'],False,dict(is_success=result['success'],autoreset=False)

    def close(self):
        self._stop_worker()
        self._closed = True
