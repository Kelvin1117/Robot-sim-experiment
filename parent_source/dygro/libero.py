"""Single-process policy with a trusted local simulator subprocess, no auto-reset."""
import json
import os
import socket
import subprocess
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
import torch


class LiberoEnvironment:
    def __init__(self, output, scripts_root):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        sim_python = '/common/home/cs2202/miniconda3/envs/libero/bin/python'
        query = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=30).splitlines()
        if len(query) != 1:
            raise ValueError('Request exactly one GPU for this single-environment implementation')
        egl_script = '/common/home/cs2202/VLARLKit/experiments/github_official_ppo_benchmark/egl_devices.py'
        env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
        devices = json.loads(subprocess.check_output([sim_python, egl_script], env=env, text=True, timeout=30))
        matches = [d['egl_index'] for d in devices if d.get('uuid') == query[0].strip()]
        if len(matches) != 1:
            raise ValueError('Cannot uniquely map allocated GPU to EGL')
        env.update(MUJOCO_EGL_DEVICE_ID=str(matches[0]), OFFICIAL_OUTPUT=str(self.output),
                   OFFICIAL_RUNTIME=str(self.output/'runtime'))
        left, right = socket.socketpair()
        self._pipe = Connection(left.detach())
        self._log = (self.output/'simulator.log').open('w')
        self._process = subprocess.Popen([sim_python, '-u', '-B', str(Path(scripts_root)/'libero_bridge.py'), str(right.fileno())],
                                         pass_fds=(right.fileno(),), env=env, stdout=self._log, stderr=subprocess.STDOUT)
        right.close()
        self._steps = 0

    def request(self, value):
        self._pipe.send(value)
        if not self._pipe.poll(180):
            raise TimeoutError('LIBERO simulator timed out')
        result = self._pipe.recv()
        if not result['ok']:
            raise RuntimeError(result['error'])
        return result

    def reset(self, task, initial_state):
        result = self.request(dict(command='reset', task_id=task, episode=initial_state, seed=0))
        self._steps = 0
        return result['obs'], dict(instruction=result['instruction'])

    def step(self, action):
        result = self.request(dict(command='step', action=np.asarray(action).tolist()))
        self._steps += 1
        success = result['success']
        if result['simulator_done'] and not success:
            raise RuntimeError('Unexpected simulator terminal state; inspect rather than label success')
        return result['obs'], result['reward'], success, False, dict(is_success=success, autoreset=False)

    def close(self):
        if self._process.poll() is None:
            try:
                self.request(dict(command='close'))
            except Exception:
                self._process.terminate()
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        finally:
            self._pipe.close()
            self._log.close()


def prepare_live_observation(raw, instruction, preprocessor):
    from lerobot.processor.env_processor import LiberoProcessorStep
    obs = {f'observation.images.{name}': torch.tensor(raw[key].copy()).permute(2, 0, 1)[None].float()/255
           for name, key in [('image', 'agentview_image'), ('image2', 'robot0_eye_in_hand_image')]}
    obs['observation.robot_state'] = {
        'eef': {'pos':torch.tensor(raw['robot0_eef_pos'])[None], 'quat':torch.tensor(raw['robot0_eef_quat'])[None]},
        'gripper': {'qpos':torch.tensor(raw['robot0_gripper_qpos'])[None]},
    }
    return preprocessor({**LiberoProcessorStep().observation(obs), 'task':[instruction]})
