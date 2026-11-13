"""Rerun cached-BEV map inference with physically serialized temporal state.

The frozen checkpoint was selected on this validation set. These measurements
are a mechanism check, not independent-test results nor official reproduction.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,platform,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE/'src'))
from edgebev.config import CFG
from edgebev.models.head_model import HeadModel
from edgebev.models.heads import decode_map
from edgebev.metrics import map_ap
from train_heads import BEVHeadDataset,collate
from train_map import forward_seq
from budget_state_codec import encode_state,decode_state,check_codec

SETTINGS=[('legacy_fp32',1.,10,'legacy'),('full_fp16',1.,10,'uncertainty'),
          ('half_u_q10',.5,10,'uncertainty'),('quarter_u_q10',.25,10,'uncertainty'),
          ('tenth_u_q10',.1,10,'uncertainty'),('full_q3',1.,3,'uncertainty'),
          ('tenth_u_q3',.1,3,'uncertainty'),('tenth_mag_q3',.1,3,'magnitude'),
          ('tenth_rand_q3_s0',.1,3,'random'),('tenth_rand_q3_s1',.1,3,'random')]


def infer(model,b,cfg,budget,nq,mode):
    outs=[];packet=None;sizes=[];timings=[]
    for ti in range(b['bev'].shape[1]):
        start=time.perf_counter()
        if packet is None:h=q=p=None
        else:h,q,p=decode_state(packet)
        h=model.gru(h,b['bev'][:,ti])
        dx=0. if ti==0 else float(b['ego'][0,ti,0]-b['ego'][0,ti-1,0])
        logits,pts,qnew,pnew=model.map_step(model.up(h),q,p,dx,k_prop=nq)
        # Store only the queries actually carried into the next frame.
        packet=encode_state(h,b['unc'][:,ti],qnew,pnew,budget,nq,mode)
        timings.append((time.perf_counter()-start)*1000)
        sizes.append(len(packet));outs.append((logits,pts))
    return outs,sizes,timings,packet


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default=str(ROOT/'rerun_evidence'))
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--limit',type=int,default=30)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    ckpath=ROOT/'runs'/'map_crf_gru_q'/'best.pt'
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    cfg=ck['cfg'];model=HeadModel(CFG,'none',cfg['map_temporal'],cfg['query_prop'])
    model.load_state_dict(ck['model']);model.eval()
    ds=BEVHeadDataset('crf','val')
    batches=[collate([ds[i]]) for i in range(min(len(ds),args.limit))]
    checks=check_codec();print('CODEC',json.dumps(checks),flush=True)
    env={'recorded_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,
         'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),
         'processor':platform.processor(),'logical_cpu_count':os.cpu_count(),
         'cuda_available':torch.cuda.is_available(),'threads':args.threads,
         'sequence_count':len(batches),'frames_per_sequence':CFG.frames,
         'checkpoint_sha256':sha(ckpath),'checkpoint_metrics_historical':ck['metrics'],
         'split':'val (also used for historical checkpoint selection)',
         'latency_scope':'batch1 cached-BEV head plus binary encode/decode, excluding image/depth/LSS and metric decode',
         'payload_scope':'24-byte header + BEV FP16 + uint16 cell indices + FP16 query embeddings and polyline points',
         'codec_checks':checks}
    (out/'environment.json').write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=[{'file':str(ckpath.relative_to(ROOT)),'sha256':sha(ckpath)}]
    for parent in [ROOT/'data'/'synthetic'/'val',ROOT/'data'/'bev'/'crf'/'val']:
        manifest.extend({'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted(parent.glob('*.npz'))[:args.limit])
    (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows=[]
    with torch.inference_mode():
        for name,budget,nq,mode in SETTINGS:
            seed=1 if name.endswith('s1') else 0
            torch.manual_seed(seed)
            # Warm up. Restore the seed so random selection is reproducible.
            if mode=='legacy':forward_seq(model,batches[0],cfg)
            else:infer(model,batches[0],cfg,budget,nq,mode)
            torch.manual_seed(seed)
            preds=[];gts=[];sizes=[];times=[]
            for bi,b in enumerate(batches):
                if mode=='legacy':
                    t=time.perf_counter();outs=forward_seq(model,b,cfg)
                    times.extend([(time.perf_counter()-t)*1000/CFG.frames]*CFG.frames)
                else:
                    outs,ss,tt,packet=infer(model,b,cfg,budget,nq,mode)
                    sizes.extend(ss);times.extend(tt)
                    if bi==0:(out/(name+'.bin')).write_bytes(packet)
                for ti,(logits,pts) in enumerate(outs):
                    preds.append(decode_map(logits,pts,CFG)[0])
                    mask=b['map_mask'][0,ti].numpy()>0
                    gts.append([(int(c),p) for c,p,ok in zip(b['map_cls'][0,ti].numpy(),b['map_line'][0,ti].numpy(),mask) if ok])
            m=map_ap(preds,gts)
            row={'setting':name,'budget':budget,'query_count':nq,'mode':mode,'seed':seed,
                 'map_mAP':m['mAP'],'AP@0.5':m['AP@0.5'],'AP@1':m['AP@1'],'AP@1.5':m['AP@1.5'],
                 'payload_bytes':max(sizes) if sizes else '',
                 'payload_KiB':max(sizes)/1024 if sizes else '',
                 'head_codec_median_ms':float(np.median(times)),
                 'head_codec_p95_ms':float(np.percentile(times,95)),
                 'frames':len(preds)}
            rows.append(row);print(json.dumps(row),flush=True)
            with (out/'rerun_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
    (out/'rerun_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',out,flush=True)


if __name__=='__main__':main()
