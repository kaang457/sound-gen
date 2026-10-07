"""Train residual adapter on cached real latents; no quality claims from MSE."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.foley.prepare_cache import validate_manifest
from src.foley.controls import ControlSpec,ControlBank
from src.foley.adapter import EpsilonAdapter,training_loss
from src.foley.backbone import load_backbone

def train(root,index_path,steps=1000,device='cpu',seed=42):
    if not isinstance(steps,int) or steps<1:raise ValueError('Positive steps required')
    root=Path(root);rows=validate_manifest(json.loads(Path(index_path).read_text()))
    splits={s:[r for r in rows if r['split']==s] for s in ['train','validation','test']}
    if len({r['recording_id'] for r in splits['train']})<2 or any(not splits[s] for s in ['validation','test']):raise ValueError('Need at least two distinct train recordings and nonempty validation/test; single-recording smoke cannot become a trained controller')
    # Reject byte-identical recordings crossing split even if IDs were renamed.
    seen={}
    for row in rows:
        h=row['sha256']
        if h in seen and seen[h]!=row['split']:raise ValueError('Byte-identical audio leakage')
        seen[h]=row['split']
    torch.set_num_threads(4);torch.manual_seed(seed);model,_=load_backbone(root,device,seed)
    config=json.loads((root/'configs/foley_controls.json').read_text());bank=ControlBank([ControlSpec(**x) for x in config['controls']],config['width']);adapter=EpsilonAdapter(bank).to(device)
    opt=torch.optim.Adam(adapter.parameters(),lr=1e-4);history=[];start=time.monotonic();trained_controls=set()
    for step in range(steps):
        item=splits['train'][int(torch.randint(len(splits['train']),(1,)).item())]
        cached=torch.load(item['cache_path'],map_location=device,weights_only=True);request=cached['request'];trained_controls.update(request['controls'])
        # Drop complete control sets on 10% of examples to expose absent controls.
        if torch.rand(()).item()<.1:request={'family':request['family'],'controls':{}}
        opt.zero_grad();loss,metrics=training_loss(model,adapter,cached['latent'],cached['condition'],[request],cached['conditioning_duration_s'])
        loss.backward();opt.step()
        if step%100==0 or step==steps-1:history.append({'step':step,'loss':float(loss.detach()),**metrics})
    destination=root/'models/foley';destination.mkdir(parents=True,exist_ok=True)
    torch.save({'status':'trained','trained_controls':sorted(trained_controls),'registry':bank.manifest(),'state_dict':adapter.state_dict(),'steps':steps,'seed':seed,'quality_evaluation':'not measured'},destination/'adapter.pt')
    report={'status':'adapter_training_completed','steps':steps,'seed':seed,'seconds':time.monotonic()-start,'device':device,'history':history,'trained_controls':sorted(trained_controls),'recording_counts':{s:len({r['recording_id'] for r in a}) for s,a in splits.items()},'validation_status':'not run by this training script','test_status':'not run','quality_evaluation':'not measured','scope':'Training pipeline only. Validation/test inference, model selection and learned control accuracy require separate evaluation; training completed does not mean usable Foley control.'}
    folder=root/'experiments/architecture';folder.mkdir(parents=True,exist_ok=True);(folder/'adapter_training.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--steps',type=int,default=1000);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args();print(json.dumps(train(ROOT,a.index,a.steps,a.device),indent=2))
