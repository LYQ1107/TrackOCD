#!/usr/bin/env python3
"""Train one bounded Phase81P association fold and emit resumable evidence."""
from __future__ import annotations
import argparse, datetime, json, os, random, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name('.'+path.name+'.tmp'); tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'); os.replace(tmp,path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--tag',default='formal'); ap.add_argument('--route',default='default',help='isolated output namespace; default preserves the original Phase81P layout'); ap.add_argument('--data-root',default='/data2/usr_for_deadline/trackocd_phase81p/data'); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--max-steps',type=int,default=None); ap.add_argument('--batch-size',type=int,default=256); ap.add_argument('--seed',type=int,default=8101); ap.add_argument('--balance-positive',action='store_true',help='deterministically balance TRAIN association vs NEW examples'); args=ap.parse_args()
    import torch
    from src.iclr27_phase81p.association import AssociationTransformer
    torch.manual_seed(args.seed+args.fold); np.random.seed(args.seed+args.fold); random.seed(args.seed+args.fold)
    data_dir=Path(args.data_root); fit_path=data_dir/f'fold{args.fold}.npz'; val_path=data_dir/f'fold{args.fold}_val.npz'
    if not fit_path.is_file(): raise FileNotFoundError(fit_path)
    fit=np.load(fit_path); x=np.asarray(fit['x'],dtype=np.float32); y=np.asarray(fit['y'],dtype=np.int64)
    val=np.load(val_path) if val_path.is_file() else None; vx=np.asarray(val['x'],dtype=np.float32) if val is not None else np.zeros((0,9,16),np.float32); vy=np.asarray(val['y'],dtype=np.int64) if val is not None else np.zeros((0,),np.int64)
    device=torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith('cuda') else 'cpu'); model=AssociationTransformer().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    out=ROOT/f'outputs/iclr27_phase81p'; prefix='' if args.route in ('', 'default') else f'{args.route}/'; ckpt_dir=out/f'checkpoints/{prefix}fold{args.fold}'; metric_dir=out/f'metrics/{prefix}fold{args.fold}'; comp=out/'completion'; marker=comp/f'association_{args.route}_{args.tag}_f{args.fold}.launched'; done=comp/f'association_{args.route}_{args.tag}_f{args.fold}.done'
    if done.exists(): print(json.dumps({'status':'already_done','fold':args.fold,'tag':args.tag})); return
    marker.parent.mkdir(parents=True,exist_ok=True); marker.write_text(json.dumps({'phase':'Phase81P+','fold':args.fold,'tag':args.tag,'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'device':str(device)})+'\n')
    n=len(x); batch=max(1,min(args.batch_size,n)); steps=0; best=None; history=[]; t0=time.time()
    for epoch in range(args.epochs):
        rng=np.random.default_rng(args.seed+args.fold+epoch)
        if args.balance_positive:
            pos_idx=np.flatnonzero(y < 9); neg_idx=np.flatnonzero(y == 9)
            if len(pos_idx) and len(neg_idx):
                # Repeat TRAIN positives to match the negative count; labels,
                # candidate sets and causal order remain unchanged.
                order=rng.permutation(np.concatenate([neg_idx, np.resize(pos_idx, len(neg_idx))]))
            else:
                order=rng.permutation(n)
        else:
            order=rng.permutation(n)
        model.train()
        for start in range(0,n,batch):
            idx=order[start:start+batch]; xb=torch.from_numpy(x[idx]).to(device); yb=torch.from_numpy(y[idx]).to(device)
            pair,new=model.score_candidates(xb); logits=torch.cat([pair,new.unsqueeze(1)],dim=1); loss=torch.nn.functional.cross_entropy(logits,yb)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); steps+=1
            if args.max_steps is not None and steps>=args.max_steps: break
        model.eval(); metrics={'epoch':epoch+1,'step':steps,'train_loss':float(loss.detach().cpu()),'val':evaluate(model,vx,vy,device),'wall_seconds':time.time()-t0}
        history.append(metrics); metric_dir.mkdir(parents=True,exist_ok=True); atomic_json(metric_dir/f'step_{steps:06d}.json',metrics)
        state={'schema_version':'phase81p.association_checkpoint.v1','fold':args.fold,'tag':args.tag,'epoch':epoch+1,'step':steps,'seed':args.seed+args.fold,'model':model.state_dict(),'optimizer':opt.state_dict(),'metrics':metrics}
        ckpt_dir.mkdir(parents=True,exist_ok=True); tmp=ckpt_dir/f'.step_{steps:06d}.pt.tmp'; torch.save(state,tmp); os.replace(tmp,ckpt_dir/f'step_{steps:06d}.pt'); torch.save(state,ckpt_dir/'latest.pt')
        criterion='balanced_accuracy' if args.balance_positive else 'accuracy'
        if best is None or metrics['val'].get(criterion,-1)>best['val'].get(criterion,-1): best=metrics; torch.save(state,ckpt_dir/'best.pt')
        if args.max_steps is not None and steps>=args.max_steps: break
    summary={'schema_version':'phase81p.association_train_summary.v2','phase':'Phase81P+','fold':args.fold,'tag':args.tag,'route':args.route,'data_root':str(data_dir),'status':'PASS_TRAINED','steps':steps,'epochs':len(history),'fit_examples':n,'val_examples':len(vx),'device':str(device),'parameters':sum(p.numel() for p in model.parameters()),'history':history,'best':best,'wall_seconds':time.time()-t0,'sampling':{'balance_positive':bool(args.balance_positive),'selection_metric':'balanced_accuracy' if args.balance_positive else 'accuracy'},'forbidden_inputs':['category_id','track_id','physical_id','semantic_id','future','held_gt']}
    atomic_json(metric_dir/'summary.json',summary); tmp=done.with_name('.'+done.name+'.tmp'); tmp.write_text('complete\n'); os.replace(tmp,done); print(json.dumps(summary,indent=2))

def evaluate(model,x,y,device):
    import torch
    if len(x)==0:return {'examples':0,'accuracy':None,'pair_accuracy':None,'positive_accuracy':None,'new_accuracy':None,'balanced_accuracy':None,'new_rate':None,'new_target_rate':None}
    with torch.no_grad():
        logits,new=model.score_candidates(torch.from_numpy(x).to(device)); all_logits=torch.cat([logits,new.unsqueeze(1)],dim=1); pred=all_logits.argmax(1).cpu().numpy()
    y=np.asarray(y); pos=(y<9); new=(y==9); pos_acc=float(((pred==y)&pos).sum()/max(1,pos.sum())); new_acc=float(((pred==y)&new).sum()/max(1,new.sum())); return {'examples':int(len(y)),'accuracy':float((pred==y).mean()),'pair_accuracy':pos_acc,'positive_accuracy':pos_acc,'new_accuracy':new_acc,'balanced_accuracy':float(0.5*(pos_acc+new_acc)),'new_rate':float((pred==9).mean()),'new_target_rate':float(new.mean())}

if __name__=='__main__': main()
