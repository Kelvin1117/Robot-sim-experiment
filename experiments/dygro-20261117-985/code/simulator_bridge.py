"""Python 3.8 LIBERO process; policy/learning stays in native OpenPI environment."""
import os
import json
import random
import sys
import traceback
from pathlib import Path
from multiprocessing.connection import Connection
import numpy as np

REPO=Path('/common/home/researcher/VLARLKit')
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
                # Construction randomizes simulator/controller state too. A fresh
                # episode must not inherit model or solver state from its predecessor.
                if env is not None:
                    env.close()
                    env=None
                random.seed(req['seed'])
                np.random.seed(req['seed'])
                bddl=REPO/'third_party/LIBERO/libero/libero/bddl_files'/task.problem_folder/task.bddl_file
                env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=req['camera_resolution'],camera_widths=req['camera_resolution'])
                current=req['task']
                random.seed(req['seed'])
                np.random.seed(req['seed'])
                env.seed(req['seed'])
                env.reset()
                mode = os.environ.get('DYGRO_RENDER_MODE', 'default')
                if mode != 'default':
                    context = env.sim._render_context_offscreen
                    context.gl_ctx.make_current()
                    from OpenGL import GL
                    if mode == 'no_dither':
                        GL.glDisable(GL.GL_DITHER)
                    elif mode in ('no_msaa', 'msaa4'):
                        env.sim.model.vis.quality.offsamples = 4 if mode == 'msaa4' else 0
                        context.con.free()
                        context._set_mujoco_context_and_buffers()
                    else:
                        raise ValueError(mode)
                raw=env.set_init_state(suite.get_task_init_states(req['task'])[req['initial']])
                for _ in range(req['warmup_steps']):raw,_,_,_=env.step([0.]*6+[-1.])
                env.sim._render_context_offscreen.gl_ctx.make_current()
                from OpenGL import GL
                renderer = {name: GL.glGetString(key).decode() for name, key in
                            [('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER), ('version', GL.GL_VERSION)]}
                renderer['offsamples'] = int(env.sim.model.vis.quality.offsamples)
                renderer['context_offsamples'] = int(env.sim._render_context_offscreen.con.offSamples)
                if mode == 'msaa4' and (renderer['offsamples'] != 4 or renderer['context_offsamples'] < 2):
                    raise RuntimeError('MSAA framebuffer was not enabled: ' + repr(renderer))
                if 'NVIDIA' not in renderer['vendor']:
                    raise RuntimeError('Expected NVIDIA GPU rendering: ' + repr(renderer))
                (Path(os.environ['OFFICIAL_OUTPUT'])/'gl_renderer.json').write_text(json.dumps(renderer, indent=2)+'\n')
                result=dict(obs=observed(raw),instruction=task.language,control_hz=env.env.control_freq)
            elif req['command']=='dataset_state':
                # Diagnostic only: reconstruct a stored simulation state without an action.
                raw=env.set_init_state(np.asarray(req['state']))
                result=dict(obs=observed(raw))
            else:
                raw,reward,done,info=env.step(req['action'])
                result=dict(obs=observed(raw),success=bool(env.check_success()),simulator_done=bool(done))
            if 'obs' in result and os.environ.get('DYGRO_CAPTURE_STATE') == '1':
                state = {}
                for key in ['qpos', 'qvel', 'qacc_warmstart', 'ctrl']:
                    state['sim_'+key] = getattr(env.sim.data, key).copy()
                for key in ['body_pos', 'body_quat']:
                    state['model_'+key] = getattr(env.sim.model, key).copy()
                result['diagnostic_state'] = state
            pipe.send(dict(ok=True,**result))
        except BaseException:pipe.send(dict(ok=False,error=traceback.format_exc()))
finally:
    if env is not None:env.close()
    pipe.close()
