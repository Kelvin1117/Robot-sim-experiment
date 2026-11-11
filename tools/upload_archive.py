#!/usr/bin/env python3
"""Stream an experiment tar.zst into verified GitHub Release assets."""
import concurrent.futures,hashlib,json,os,re,subprocess,sys,tarfile,threading,time
from pathlib import Path
GH='/common/users/cs2202/upload-tools/gh'
REPO='Kelvin1117/Robot-sim-experiment'
TAG='retention-20261110'
WORK=Path('/common/users/cs2202/robot-experiment-assets')
TMP=Path('/tmp/cs2202-robot-experiment-assets')
WORK.mkdir(exist_ok=True);TMP.mkdir(exist_ok=True);TMP.chmod(0o700)
root=Path('/common/users/cs2202/dygro_pi05_phase012_20261106')
parent=root/'oneshot_paper_v2'
sources=[root/'oneshot_retention',parent/'runs/paper-v2-20261108T174348Z',root/'oneshot_openpi/data/libero_goal',Path('/common/users/cs2202/models/pi05_libero_one_shot_pytorch'),Path('/common/users/cs2202/models/pi05_libero_one_shot/5000'),Path('/common/home/cs2202/VLARLKit/model_backends/openpi'),Path('/common/home/cs2202/VLARLKit/model_backends/openpi/.venv'),Path('/common/home/cs2202/VLARLKit/third_party/LIBERO'),Path('/common/home/cs2202/miniconda3/envs/libero'),Path('/common/users/cs2202/vlarlkit/config/libero'),Path('/common/users/cs2202/vlarlkit/cache/openpi/big_vision/paligemma_tokenizer.model'),Path('/common/home/cs2202/VLARLKit/experiments/author_sft_goal20'),Path('/common/home/cs2202/VLARLKit/experiments/github_official_ppo_benchmark/egl_devices.py'),Path('/common/home/cs2202/VLARLKit/vlarlkit'),Path('/common/home/cs2202/VLARLKit/env_clients')]
sources += [p for p in parent.iterdir() if p.name not in ('runs','data')]
sources += [p for p in Path('/common/home/cs2202/VLARLKit').iterdir() if p.is_file() and (p.name.startswith(('LICENSE','NOTICE')) or p.name in ('pyproject.toml','setup.py','setup.cfg'))]

def produce():
    seen=set();excluded=[];count=0;total=0;current=None
    manifest=(WORK/'archive_files.jsonl').open('w')
    class HashReader:
        def __init__(self,f):self.f=f;self.digest=hashlib.sha256()
        def read(self,n=-1):
            data=self.f.read(n);self.digest.update(data);return data
    class RecordedTar(tarfile.TarFile):
        def addfile(self,info,fileobj=None):
            row={'path':info.name,'size':info.size,'type':info.type.decode('ascii'),'linkname':info.linkname}
            wrapped=HashReader(fileobj) if fileobj is not None else None
            super().addfile(info,wrapped)
            if wrapped:row['sha256']=wrapped.digest.hexdigest()
            manifest.write(json.dumps(row)+'\n')
    def filt(info):
        nonlocal count,total
        p=Path(info.name)
        if info.name in seen:return None
        if any(x in p.parts for x in ('.git','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache')) or p.name.lower().startswith('readme') or p.name in ('.env','.git-credentials','hosts.yml','token','stored_tokens','.netrc','id_rsa','id_ed25519') or p.name.startswith('.env.'):
            excluded.append(info.name);return None
        if current.name=='openpi' and '.venv' in p.parts:return None
        seen.add(info.name);count+=1;total+=info.size
        info.uid=0;info.gid=0;info.uname='';info.gname=''
        return info
    with RecordedTar.open(fileobj=sys.stdout.buffer,mode='w|',dereference=False,copybufsize=1024*1024) as archive:
        for current in sources:
            if not current.exists():raise FileNotFoundError(current)
            print('Archiving '+str(current),file=sys.stderr,flush=True)
            archive.add(current,arcname=str(current).lstrip('/'),filter=filt)
    manifest.close()
    (WORK/'archive_source_summary.json').write_text(json.dumps({'entries':count,'logical_regular_bytes':total,'sources':[str(p) for p in sources],'excluded':excluded},indent=2)+'\n')

def gh(*args):
    p=subprocess.run([GH,*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if p.returncode:raise RuntimeError('GitHub command failed: '+p.stderr[-3000:])
    return p.stdout

def run():
    release=next(r for r in json.loads(gh('api',f'repos/{REPO}/releases?per_page=100')) if r['tag_name']==TAG)
    if not release['draft']:raise RuntimeError('Refusing to alter a published release')
    if release['assets']:raise RuntimeError('Release already has assets; inspect before restarting')
    producer_log=(WORK/'producer.log').open('wb')
    producer=subprocess.Popen([sys.executable,__file__,'--produce'],stdout=subprocess.PIPE,stderr=producer_log)
    compressor=subprocess.Popen(['/common/home/cs2202/miniconda3/bin/zstd','-1','-T4','-c'],stdin=producer.stdout,stdout=subprocess.PIPE,stderr=producer_log)
    producer.stdout.close()
    validator=subprocess.Popen(['/common/home/cs2202/miniconda3/bin/zstd','-t','-q','-'],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=producer_log)
    slots=threading.Semaphore(2);parts=[];futures=[];started=time.time()
    def upload(path,digest):
        try:
            last=None
            for attempt in range(4):
                try:
                    # An interrupted previous attempt may already have reached GitHub.
                    assets=json.loads(gh('api',f'repos/{REPO}/releases/{release["id"]}/assets?per_page=100'))
                    found=next((a for a in assets if a['name']==path.name),None)
                    if not found:
                        gh('release','upload',TAG,str(path),'--repo',REPO)
                        assets=json.loads(gh('api',f'repos/{REPO}/releases/{release["id"]}/assets?per_page=100'))
                        found=next((a for a in assets if a['name']==path.name),None)
                    if not found or found['size']!=path.stat().st_size or found.get('digest')!='sha256:'+digest:
                        raise RuntimeError('Remote asset size/digest mismatch: '+path.name)
                    row={'file':path.name,'size':path.stat().st_size,'sha256':digest,'asset_id':found['id'],'url':found['browser_download_url']}
                    parts.append(row)
                    with (WORK/'uploaded_parts.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
                    print('Verified uploaded '+path.name+' '+str(row['size'])+' bytes',flush=True)
                    path.unlink()
                    return row
                except Exception as e:
                    last=e
                    if attempt<3:time.sleep(2**attempt)
            raise last
        finally:slots.release()
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        index=0
        while True:
            for f in futures:
                if f.done():f.result()
            slots.acquire()
            path=TMP/f'experiment.tar.zst.part-{index:03d}'
            digest=hashlib.sha256();size=0
            with path.open('wb') as output:
                while size<768*1024*1024:
                    block=compressor.stdout.read(min(8*1024*1024,768*1024*1024-size))
                    if not block:break
                    validator.stdin.write(block)
                    output.write(block);digest.update(block);size+=len(block)
            if size==0:
                path.unlink();slots.release();break
            futures.append(pool.submit(upload,path,digest.hexdigest()));index+=1
        validator.stdin.close()
        results=[f.result() for f in futures]
        statuses=[producer.wait(),compressor.wait(),validator.wait()]
        if any(statuses):raise RuntimeError('Archive pipeline failed: '+str(statuses))
        summary=json.loads((WORK/'archive_source_summary.json').read_text())
        summary.update({'repository':REPO,'release_tag':TAG,'format':'Concatenate numbered parts in lexical order; decode zstd; extract tar','archive_paths':'Original filesystem paths without leading slash','parts':sorted(results,key=lambda x:x['file']),'compressed_bytes':sum(p['size'] for p in results),'zstd_integrity_verified':True,'uploaded':True,'elapsed_seconds':time.time()-started})
        (WORK/'asset_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
        (WORK/'SHA256SUMS').write_text(''.join(p['sha256']+'  '+p['file']+'\n' for p in summary['parts']))
        print(json.dumps({'complete':True,'entries':summary['entries'],'parts':len(parts),'compressed_bytes':summary['compressed_bytes']}),flush=True)
    except BaseException:
        producer.kill();compressor.kill();validator.kill();raise
    finally:
        pool.shutdown(wait=True);producer_log.close()

if __name__=='__main__':
    produce() if '--produce' in sys.argv else run()
