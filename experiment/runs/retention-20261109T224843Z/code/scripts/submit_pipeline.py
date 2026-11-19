"""Verify immutable code, GPU smoke, capacity and four afterok jobs; then exit."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from snapshot import PORT,runtime_manifest,snapshot


def command(args):
    return subprocess.check_output(args,text=True,timeout=60).strip()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def quota_free_bytes():
    # NFS quota, not df's entire shared filesystem capacity.
    raw=command(['quota','-w'])
    matches=re.findall(r'eternal\.lcsr\.rutgers\.edu:/common/users\s+(\d+)\*?\s+(\d+)\s+(\d+)',raw)
    if len(matches)!=1:raise ValueError('Cannot verify personal users quota')
    used,soft,hard=map(int,matches[0])
    return (min(soft,hard)-used)*1024


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke-run',type=Path,required=True)
    args=parser.parse_args();smoke=args.smoke_run.resolve()
    if (smoke/'SMOKE_PASS').read_text().strip()!='PASS':raise ValueError('GPU integration smoke has not passed')
    if runtime_manifest(PORT)!=json.loads((smoke/'source_manifest.json').read_text()):raise ValueError('Runtime changed after smoke')
    tests=(PORT/'reports/cpu_tests.txt').read_text()
    if not re.search(r'\b33 passed\b',tests) or 'FAILED' in tests:raise ValueError('Required CPU tests did not pass')
    cfg=json.loads((PORT/'configs/full.json').read_text());sc=json.loads((smoke/'config.json').read_text())
    if cfg['scope']!='formal' or cfg['fusion']!=sc['fusion'] or cfg['model']!=sc['model']:raise ValueError('Model/architecture mismatch')
    if cfg['stage1']['batch_size']!=sc['stage1']['batch_size']:raise ValueError('Formal batch size was not GPU-tested')
    if cfg['stage1']['initial_steps']*cfg['stage1']['batch_size']/cfg['stage1']['training_split_observations']<4:raise ValueError('Insufficient Stage I budget')
    gate=cfg['stage1_gate']
    if gate['minimum_overall_success_rate']<.75 or gate['minimum_per_task_success_rate']<.2 or not gate['allow_budget_extension']:raise ValueError('Do not weaken the behavioral gate')
    if len(cfg['evaluation']['seeds'])*cfg['evaluation']['episodes_per_task']*10!=600:raise ValueError('Need final 600 episodes')
    if set(cfg['stage2']['train_initial_states']) & (set(cfg['evaluation']['initial_states'])|set(gate['initial_states'])):raise ValueError('Reset state overlap')
    for stage in ['stage1','validate','stage2','evaluate']:
        records=[json.loads(l) for l in (smoke/f'{stage}_status.jsonl').read_text().splitlines()]
        if records[-1].get('exit_status')!=0:raise ValueError(f'Smoke {stage} failed')
    interface=json.loads((smoke/'live_interface.json').read_text())
    if not interface['pretrained_load_audit']['exact_parameter_equality']:raise ValueError('Original SFT weights not verified')
    free=quota_free_bytes()
    if free<14*1024**3:raise ValueError(f'Need 14GiB personal quota headroom; available {free/1024**3:.2f}GiB')
    run=snapshot(PORT/'runs'/('paper-v2-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())),PORT/'configs/full.json')
    for name in ['cpu_tests.txt','parameter_provenance.json','method_audit.md','external_dependency_hashes.json','package_versions.json','gpu_integration.json']:shutil.copy2(PORT/'reports'/name,run/name)
    write(run/'verification.json',dict(smoke=str(smoke),source_manifest=json.loads((smoke/'source_manifest.json').read_text()),live_interface=interface,personal_quota_free_bytes=free,note='Integration passed; not a trained success-rate result'))
    stages=[('stage1','train_stage1.sbatch'),('validation','validate_stage1.sbatch'),('stage2','train_stage2.sbatch'),('evaluation','evaluate_dygro.sbatch')]
    jobs={};previous=None
    try:
        for stage,script in stages:
            options=['sbatch','--parsable','--hold','--kill-on-invalid-dep=yes','--output',str(run/'logs'/f'{stage}-%j.out'),'--error',str(run/'logs'/f'{stage}-%j.err')]
            if previous:options+=['--dependency','afterok:'+previous]
            output=command(options+[str(run/script),str(run)])
            match=re.fullmatch(r'(\d+)(?:;[^\n]+)?',output)
            if not match:raise ValueError(f'Invalid sbatch result: {output}')
            job=match[1];jobs[stage]=job
            write(run/'pipeline.json',dict(run=str(run),jobs=jobs,state='held_until_verified'))
            details=command(['scontrol','show','job',job]);(run/'logs'/f'submitted-{job}.txt').write_text(details+'\n')
            if previous and f'Dependency=afterok:{previous}' not in details:raise ValueError('Incorrect dependency')
            if 'TresPerNode=gres/gpu:a100:1' not in details or 'Partition=unlimited' not in details or 'mem=96G' not in details:raise ValueError('Incorrect GPU/partition/memory')
            if not re.search(r'NumCPUs=[1-9]\d*',details) or f'Command={run/script}' not in details or f'StdOut={run}/logs/{stage}-{job}.out' not in details:raise ValueError('Incorrect CPU/launcher/log path')
            previous=job
    except BaseException as exc:
        write(run/'submission_failure.json',dict(error=str(exc),jobs=jobs,state='held; training not released'))
        raise
    for stage in ['evaluation','stage2','validation','stage1']:command(['scontrol','release',jobs[stage]])
    states={}
    for stage,job in jobs.items():
        details=command(['scontrol','show','job',job]);(run/'logs'/f'confirmed-{job}.txt').write_text(details+'\n')
        states[stage]=re.search(r'JobState=(\S+)',details).group(1)
    result=dict(run=str(run),jobs=jobs,state='submitted',states_at_submission=states,smoke=str(smoke),dependencies={jobs[b]:'afterok:'+jobs[a] for a,b in [('stage1','validation'),('validation','stage2'),('stage2','evaluation')]})
    write(run/'pipeline.json',result);write(PORT/'reports/latest_pipeline.json',result)
    root=PORT.parent;status=root/'TRAINING_STATUS.zh-TW.md'
    if status.exists():shutil.copy2(status,root/'recovery'/f'before_{run.name}.md')
    lines=['# DyGRO-VLA → π0.5 One-Shot：v2 已提交','',
        '修正後的新Slurm訓練鏈已提交；訓練與600集評估尚未完成。核對提交後結束Codex，不等待訓練。','',f'Run: `{run}`','',
        '| 階段 | Job ID | 提交時狀態 | 依賴 |','|---|---|---|---|']
    previous=None
    for stage,job in jobs.items():
        lines.append(f"| {stage} | {job} | {states[stage]} | {'afterok:'+previous if previous else '無'} |")
        previous=job
    lines+=['',f"Stage I：{cfg['stage1']['initial_steps']} steps（約4.1 epoch equivalents）；50集base-policy gate未過時，最多續訓至{cfg['stage1']['steps']} steps（約8 epochs）。整體≥75%且每task≥20%才寫validated.json；否則非零退出並阻止Stage II。這是明示工程驗證門檻，非論文公式。",'',
        'Stage II：完整2000 episodes；8 experts/top2、SAC/MoRR、4 critics+Cal、InfoNCE/LB、雙replay與difficulty sampling。正式600集評估：10tasks×20集×3seeds。','',
        '每job：A100×1、RAM96GiB；CPU依叢集政策不指定數量（allocation見logs）；Stage I/Validation/Stage II各72h上限，evaluation12h。已測A100-PCIE-40GB。','',
        '原始SFT baseline沒有重跑；權重是唯讀引用，811個stored tensors已逐一確認精確相等。','',
        'Checkpoint：latest.pt保留optimizer/scheduler/RNG（Stage II另有replays/targets）；best.pt只保留推論必需參數；stage1/validated.pt綁定通過環境驗證的權重hash。','',
        f'方法稽核：`{PORT}/reports/method_audit.md`；所有參數來源與實驗假設：run/parameter_provenance.json。骨幹接線、PaliGemma凍結、latent noise和BC/filter的未公開細節均有明示假設，不能宣稱完全忠實重現或已恢復成功率。','',
        f'查詢：`squeue -j {",".join(jobs.values())} -o "%.18i %.24j %.12T %.45E %R"`','',
        f'完成後結果：`{run}/evaluation/summary.json`。若中途失敗，檢查logs與stage1/gate-*.json，不得跳過Stage I驗證。']
    status.write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))
    print(command(['squeue','-j',','.join(jobs.values()),'-o','%.18i %.25j %.12T %.45E %R']))

if __name__=='__main__':main()
