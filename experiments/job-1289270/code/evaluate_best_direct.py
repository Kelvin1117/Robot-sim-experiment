import json
import sys
from pathlib import Path
import torch
import recovery_run as rr

run = Path(sys.argv[1]).resolve()
settings = json.loads((run / 'config.json').read_text())
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
rr.seed_all(settings['seed'])
config, adapter, pre, data, learner, _, _ = rr.prepare(settings, run, False)
learner.actor.correction.load_state_dict(rr.load(run / 'best.pt')['correction'])
learner.actor.enabled = True
env = rr.original.Environment(run / 'best_direct_simulator', settings=config['environment'])
try:
    print('Evaluating best.pt directly; correction enabled; repeatability check skipped', flush=True)
    result = rr.evaluate(config, settings, adapter, pre, learner.actor, env,
                         rr.original.Log(run / 'best_direct_episodes.jsonl'),
                         settings['evaluation_seeds'], settings['evaluation_initial_states'])
    result.update(checkpoint_sha256=rr.digest(run / 'best.pt'), repeatability_checked=False)
    rr.atomic_json(run / 'best_direct_summary.json', result)
    print(json.dumps(result), flush=True)
finally:
    env.close()
    data.close()
