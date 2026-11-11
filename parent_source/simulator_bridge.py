"""Python 3.8 LIBERO process; policy/learning stays in native OpenPI environment."""
import os
import sys
import traceback
from pathlib import Path
from multiprocessing.connection import Connection
import numpy as np

REPO=Path('/common/home/cs2202/VLARLKit')
sys.path[:0]=[str(REPO/'third_party/LIBERO'),str(REPO/'experiments/author_sft_goal20')]
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

pipe=Connection(int(sys.argv[1]));suite=benchmark.get_benchmark_dict()['libero_goal'](task_order_index=0)
env=None;current=None


def observed(raw):
    return {k:raw[k] for k in ['agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']}


try:
    while True:
        req=pipe.recv()
        try:
            if req['command']=='close':
                pipe.send(dict(ok=True));break
            if req['command']=='reset':
                task=suite.get_task(req['task'])
                if current!=req['task']:
                    if env is not None:env.close()
                    bddl=REPO/'third_party/LIBERO/libero/libero/bddl_files'/task.problem_folder/task.bddl_file
                    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=req['camera_resolution'],camera_widths=req['camera_resolution'])
                    env.seed(req['seed']);current=req['task']
                env.reset();raw=env.set_init_state(suite.get_task_init_states(req['task'])[req['initial']])
                for _ in range(req['warmup_steps']):raw,_,_,_=env.step([0.]*6+[-1.])
                result=dict(obs=observed(raw),instruction=task.language,control_hz=env.env.control_freq)
            elif req['command']=='dataset_state':
                # Diagnostic only: reconstruct a stored simulation state without an action.
                raw=env.set_init_state(np.asarray(req['state']))
                result=dict(obs=observed(raw))
            else:
                raw,reward,done,info=env.step(req['action'])
                result=dict(obs=observed(raw),success=bool(env.check_success()),simulator_done=bool(done))
            pipe.send(dict(ok=True,**result))
        except BaseException:pipe.send(dict(ok=False,error=traceback.format_exc()))
finally:
    if env is not None:env.close()
    pipe.close()
