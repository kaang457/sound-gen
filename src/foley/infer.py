"""Structured event request -> pretrained audio baseline; trained adapter optional.

Never silently ignore numeric controls. The architecture smoke checkpoint does
not qualify as a trained control adapter. Prompt strings are not accepted.
"""
import argparse
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
import soundfile as sf
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.foley.controls import ControlSpec,ControlBank
from src.foley.adapter import EpsilonAdapter
from src.foley.backbone import load_backbone

class TrainingRequiredError(RuntimeError):pass

def render(root,request,device='auto',output_name='foley_request',adapter_path=None):
    root=Path(root).resolve()
    allowed={'event_id','family','duration_s','seed','steps','controls'}
    if set(request)-allowed or not {'event_id','family','duration_s'}.issubset(request):raise ValueError('Invalid structured request; text prompts are not supported')
    if not output_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in output_name):raise ValueError('Invalid output stem')
    duration=request['duration_s'];steps=request.get('steps',50)
    if isinstance(duration,bool) or not isinstance(duration,(float,int)) or not np.isfinite(duration) or not 2.5<=duration<=10 or not isinstance(steps,int) or steps<2:raise ValueError('Require duration 2.5–10 s and steps >= 2')
    seed=request.get('seed',42)
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**32:raise ValueError('Invalid seed')
    event=request['event_id']
    if not isinstance(event,str) or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in event):raise ValueError('Invalid event identifier')
    prototype=json.loads((root/'experiments/architecture'/f'{event}_prototype.json').read_text())
    if prototype['event_id']!=event or prototype.get('family',event)!=request['family']:raise ValueError('Prototype event/family mismatch')
    registry=json.loads((root/'configs/foley_controls.json').read_text())
    bank=ControlBank([ControlSpec(**x) for x in registry['controls']],registry['width'])
    bank([request],int(duration*25.6),duration)  # Validate before expensive model load.
    adapter=None
    if request.get('controls'):
        if adapter_path is None:raise TrainingRequiredError('Numeric control schema exists, but a Foley-trained adapter is required. The two-step smoke is not a trained controller.')
        state=torch.load(adapter_path,map_location='cpu',weights_only=True)
        if state.get('status')!='trained' or set(request['controls'])-set(state.get('trained_controls',[])):raise TrainingRequiredError('Checkpoint does not declare training for the requested controls')
        if state.get('registry')!=bank.manifest():raise ValueError('Adapter registry mismatch')
        adapter=EpsilonAdapter(bank);adapter.load_state_dict(state['state_dict'],strict=True)
    torch.set_num_threads(4)
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if device not in ('cuda','cpu'):raise ValueError('Invalid device')
    model,_=load_backbone(root,device,request.get('seed',42));model.latent_t_size=int(duration*25.6)
    condition=torch.tensor(prototype['condition'],device=device)
    if adapter is not None:
        adapter.eval().to(device);original=model.apply_model
        def controlled(x,t,cond,return_ids=False):
            if return_ids:raise ValueError('Unsupported sampler output')
            if len(x)==1:requests=[request]
            elif len(x)==2:requests=[{'family':request['family'],'controls':{}},request]
            else:raise ValueError('Single-event CFG batch only')
            # Native codec grid extends beyond requested audio duration; hold
            # curve endpoints over the tail that is cropped after decoding.
            import copy
            requests=copy.deepcopy(requests);codec_duration=x.shape[2]*.04
            for row in requests:
                for name,value in row.get('controls',{}).items():
                    if bank.specs[name].kind=='curve':
                        if codec_duration>duration:
                            value['times_s'].append(codec_duration);value['values'].append(value['values'][-1])
                        else:value['times_s'][-1]=codec_duration
            return original(x,t,cond)+adapter(x,t,requests,codec_duration,cond)
        model.apply_model=controlled
    torch.manual_seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        unconditional=model.cond_stage_model.get_unconditional_condition(1)
        latent,_=model.sample_log(cond=condition,batch_size=1,ddim=True,ddim_steps=steps,unconditional_guidance_scale=2.5,unconditional_conditioning=unconditional,eta=0.)
        output=np.asarray(model.mel_spectrogram_to_waveform(model.decode_first_stage(latent))).reshape(-1).astype(np.float32)[:round(duration*16000)]
    if len(output)!=round(duration*16000) or not np.isfinite(output).all():raise ValueError('Invalid output audio')
    folder=root/'results/audio/foley';folder.mkdir(parents=True,exist_ok=True);path=folder/(output_name+'.wav');sf.write(path,output,16000,subtype='FLOAT')
    report={'request':request,'device':device,'user_text_prompt':None,'runtime_reference_wav_required':False,'positive_condition':'cached audio exemplar embedding','prototype_support_count':prototype['support_count'],'prototype_sha256':hashlib.sha256((root/'experiments/architecture'/f'{event}_prototype.json').read_bytes()).hexdigest(),'numeric_controls_applied':adapter is not None,'quality_evaluation':'not measured','adapter_checkpoint_sha256':hashlib.sha256(Path(adapter_path).read_bytes()).hexdigest() if adapter is not None else None,'output_wav_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'output_peak':float(abs(output).max()),'output_rms':float(np.sqrt(np.mean(output.astype(np.float64)**2)))}
    dest=root/'experiments/architecture';dest.mkdir(parents=True,exist_ok=True);(dest/(output_name+'_run.json')).write_text(json.dumps(report,indent=2));return path,report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--device',default='auto');p.add_argument('--output-name',default='foley_request');p.add_argument('--adapter',type=Path);a=p.parse_args();path,report=render(ROOT,json.loads(a.request.read_text()),a.device,a.output_name,a.adapter);print(json.dumps(report,indent=2))
