"""Restore original Orbax on CPU, map every tensor, compare existing conversion."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from safetensors import safe_open
import openpi.models.gemma

ROOT=Path(__file__).resolve().parents[1]
config=json.loads((ROOT/'configs/full.json').read_text())
source=Path('/common/users/cs2202/vlarlkit/runs/author-sft-20261024/convert_author.py')
spec=importlib.util.spec_from_file_location('historical_converter',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
initial=module.slice_initial_orbax_checkpoint(str(Path(config['model']['orbax']).parent),'float32')
paligemma=SimpleNamespace(vision_config=SimpleNamespace(hidden_size=1152,num_hidden_layers=27,num_attention_heads=16,intermediate_size=4304,patch_size=14,projection_dim=2048),
                         text_config=SimpleNamespace(hidden_size=2048,num_hidden_layers=18,num_attention_heads=8,head_dim=256,intermediate_size=16384))
vision,expert=module.slice_paligemma_state_dict(initial['paligemma_params'],paligemma)
action=module.slice_gemma_state_dict(expert,openpi.models.gemma.get_config('gemma_300m'),num_expert=1,checkpoint_dir=str(Path(config['model']['orbax']).parent),pi05=True)
mapped={**vision,**action}
for name in ['action_in_proj','action_out_proj','time_mlp_in','time_mlp_out']:
    for source_key,target_key in [('kernel','weight'),('bias','bias')]:
        value=initial['projection_params'][name][source_key]
        if isinstance(value,dict):value=value['value']
        tensor=torch.as_tensor(value.copy())
        mapped[name+'.'+target_key]=tensor.T if source_key=='kernel' else tensor
checked=0;rounded=[]
with safe_open(config['model']['weights'],framework='pt',device='cpu') as f:
    for alias,target in (f.metadata() or {}).items():
        if alias in mapped and target not in mapped:
            mapped[target]=mapped.pop(alias)
    if set(f.keys())!=set(mapped):raise ValueError(f'Mapping keys differ: {set(f.keys())^set(mapped)}')
    for name in f.keys():
        stored=f.get_tensor(name)
        expected=mapped[name]
        # The historical converter constructed OpenPI with its default BF16
        # selected-weight recipe, loaded F32, then saved a F32 model. Reproduce
        # that exact rounding, never hide arbitrary errors under a tolerance.
        keep=['vision_tower.vision_model.embeddings.patch_embedding.weight',
              'vision_tower.vision_model.embeddings.patch_embedding.bias',
              'vision_tower.vision_model.embeddings.position_embedding.weight',
              'input_layernorm','post_attention_layernorm','model.norm']
        if name.startswith('paligemma_with_expert.') and not any(k in name for k in keep):
            if not torch.equal(stored,expected):
                expected=expected.to(torch.bfloat16).float();rounded.append(name)
        if not torch.equal(stored,expected):raise ValueError(f'Orbax conversion mismatch: {name}')
        checked+=1
report=dict(status='PASS',orbax_restored=True,all_stored_tensors_equal=True,tensors=checked,
            historical_bf16_rounding=rounded,
            conversion_source=str(source),scope='weight mapping equivalence; not JAX/Torch forward numerical equivalence')
(ROOT/'reports/orbax_restore.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='historical_bf16_rounding'}),flush=True)
