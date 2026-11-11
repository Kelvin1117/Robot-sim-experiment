import json
import os
from pathlib import Path
import re
import subprocess
import sys


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def main():
    run=Path(sys.argv[1]).resolve()
    if (run/'submission.json').exists():
        raise FileExistsError('This recovery run was already submitted')
    tests=json.loads((run/'cpu_verification.json').read_text())
    if not tests['passed']:
        raise RuntimeError('CPU verification must pass before submission')
    quota=command(['quota','-w'])
    (run/'quota-before.txt').write_text(quota+'\n')
    matches=re.findall(r'eternal[^\n]*?/common/users\s+(\d+)\s+(\d+)\s+(\d+)', quota)
    if len(matches) != 1:
        raise RuntimeError('Could not verify personal users quota')
    used,soft,hard=map(int,matches[0])
    free=(hard-used)*1024
    if free < 2*1024**3:
        raise RuntimeError(f'Need 2 GiB quota headroom for correction/replay checkpoints, available {free}')
    jobs=[]
    try:
        for stage, hours in [('smoke','01:00:00'),('train','1-00:00:00'),('evaluate','06:00:00')]:
            args=['sbatch','--parsable','--hold','--kill-on-invalid-dep=yes',
                  '--job-name',f'dygro-retain-{stage}','--time',hours,
                  '--output',str(run/'logs'/f'{stage}-%j.out'),
                  '--error',str(run/'logs'/f'{stage}-%j.err')]
            if jobs:
                args += ['--dependency',f'afterok:{jobs[-1]["job_id"]}']
            args += [str(run/'job.sbatch'),str(run),stage]
            output=command(args)
            match=re.fullmatch(r'(\d+)(?:;[\w-]+)?',output)
            if not match:
                raise RuntimeError(f'Unexpected sbatch response: {output}')
            job=dict(stage=stage,job_id=int(match[1]),submission_command=args)
            jobs.append(job)
            allocation=command(['scontrol','show','job','-o',str(job['job_id'])])
            (run/'logs'/f'{stage}-{job["job_id"]}-submission.txt').write_text(allocation+'\n')
            fields=dict(re.findall(r'(\w+)=(\S+)', allocation))
            if fields.get('ReqNodeList') != 'rlab2' or fields.get('Partition') != 'unlimited':
                raise RuntimeError('Wrong node/partition')
            if 'gres/gpu:a100:1' not in allocation or fields.get('MinMemoryNode') != '96G':
                raise RuntimeError('Wrong GPU/memory allocation')
            if len(jobs)>1 and f'afterok:{jobs[-2]["job_id"]}' not in fields.get('Dependency',''):
                raise RuntimeError('Missing afterok dependency')
        submission=dict(jobs=jobs,run=str(run),node='rlab2',gpu='1 A100 per job',memory='96G',
                        quota_headroom_bytes=free,status='verified_held',
                        protocol='Each model: 10 tasks x 60 episodes; paired parent/recovered = 1200 total')
        (run/'submission.json').write_text(json.dumps(submission,indent=2)+'\n')
        # Release dependent jobs before the smoke job, so the complete chain is
        # installed before GPU work starts. No retries or monitoring loop.
        for job in reversed(jobs):
            command(['scontrol','release',str(job['job_id'])])
        submission['status']='released'
        (run/'submission.json').write_text(json.dumps(submission,indent=2)+'\n')
        status=command(['squeue','-j',','.join(str(j['job_id']) for j in jobs),'-o','%.18i %.32j %.10T %.35R'])
        (run/'queue-at-submission.txt').write_text(status+'\n')
        print(json.dumps(submission,ensure_ascii=False,indent=2))
        print(status)
    except BaseException:
        # Only jobs created by this invocation are eligible for cleanup.
        for job in jobs:
            subprocess.run(['scancel',str(job['job_id'])],check=False)
        raise


if __name__=='__main__':
    main()
