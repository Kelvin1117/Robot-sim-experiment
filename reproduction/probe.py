"""Import checks that also run under the archived Python 3.8 simulator."""
import ctypes
import json
import os
import sys

kind = sys.argv[1]
if kind == 'policy':
    import torch
    import h5py
    import openpi
    import recovery_run
    cpu_only = os.environ.get('ROBOT_CPU_ONLY') == '1'
    result = dict(python=sys.version.split()[0], torch=torch.__version__, cuda_runtime=torch.version.cuda,
                  experiment_import=True, gpu_checked=not cpu_only)
    if not cpu_only:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable: check the allocated GPU, NVIDIA driver and Container Toolkit.')
        gpu = torch.cuda.get_device_properties(0)
        result.update(gpu=gpu.name, vram_GiB=gpu.total_memory / 1024**3)
else:
    import torch
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    ctypes.CDLL('libEGL.so.1')
    ctypes.CDLL('libGL.so.1')
    suite = benchmark.get_benchmark_dict()['libero_goal'](task_order_index=0)
    states = suite.get_task_init_states(0)
    if len(states) < 50:
        raise RuntimeError('LIBERO initial states are incomplete.')
    result = dict(python=sys.version.split()[0], libero_goal_import=True,
                  task_zero_initial_states=len(states), render_libraries_loaded=True)
print('ROBOT_PROBE_JSON=' + json.dumps(result))
