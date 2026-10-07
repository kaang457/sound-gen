"""Real-recording codec/control cache. Source files are supplied by a manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import librosa
import soundfile as sf
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.foley.backbone import load_backbone

def validate_manifest(data):
    rows=data['records'];required={'recording_id','path','event_id','family','license','source_url','split'}
    groups={};events={}
    if not rows:raise ValueError('Empty manifest')
    for row in rows:
        if not required.issubset(row) or not all(isinstance(row[k],str) and row[k] for k in required):raise ValueError('Missing manifest metadata')
        if row['split'] not in ('train','validation','test'):raise ValueError('Invalid split')
        if any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in row['event_id']):raise ValueError('Invalid event_id')
        rid=row['recording_id']
        if rid in groups and groups[rid]!=row['split']:raise ValueError('Recording leakage between splits')
        groups[rid]=row['split']
        if row['event_id'] in events and events[row['event_id']]!=row['family']:raise ValueError('Event/family mismatch')
        events[row['event_id']]=row['family']
        for name in row.get('controls',{}):
            if name in ('rms_envelope','centroid_hz'):raise ValueError('Measured controls are generated, not supplied as labels')
            if not row.get('control_provenance',{}).get(name):raise ValueError('Physical/category controls require provenance')
    return rows

def prepare(root,manifest_path,device='cpu'):
    root=Path(root).resolve();manifest_path=Path(manifest_path).resolve();data=json.loads(manifest_path.read_text());rows=validate_manifest(data)
    torch.set_num_threads(4);model,_=load_backbone(root,device)
    from audioldm.audio import TacotronSTFT
    from audioldm.audio.tools import get_mel_from_wav
    import audioldm.audio.stft as legacy
    legacy.pad_center=lambda a,size:librosa.util.pad_center(a,size=size)
    legacy.librosa_mel_fn=lambda sr,n_fft,n_mels,fmin,fmax:librosa.filters.mel(sr=sr,n_fft=n_fft,n_mels=n_mels,fmin=fmin,fmax=fmax)
    stft=TacotronSTFT(1024,160,1024,64,16000,0,8000)
    output=root/'models/foley/cache';output.mkdir(parents=True,exist_ok=True);items=[]
    for i,row in enumerate(rows):
        duration=float(row.get('duration_s',5))
        if not 2.5<=duration<=10:raise ValueError('duration must be 2.5–10')
        path=Path(row['path']);path=path if path.is_absolute() else manifest_path.parent/path
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if row.get('sha256') and row['sha256']!=digest:raise ValueError('Audio hash mismatch')
        audio,sr=sf.read(path,dtype='float32',always_2d=True);audio=audio.mean(1);audio=librosa.resample(audio,orig_sr=sr,target_sr=16000)
        start=float(row.get('crop_start_s',0))
        if start<0 or not np.isfinite(start):raise ValueError('Invalid crop start')
        audio=audio[round(start*16000):]
        if len(audio)<100 or not np.isfinite(audio).all():raise ValueError('Invalid source audio')
        length=int(duration*25.6)*4*160;audio=audio[:length];audio=audio-audio.mean()
        if max(abs(audio))<1e-8:raise ValueError('Silent source')
        audio=.5*audio/(max(abs(audio))+1e-8);audio=np.pad(audio,(0,length-len(audio))).astype(np.float32)
        mel,_,_=get_mel_from_wav(audio,stft);mel=mel.T[:length//160]
        with torch.no_grad():
            latent=model.get_first_stage_encoding(model.encode_first_stage(torch.tensor(mel)[None,None].to(device))).cpu()
            condition=model.get_learned_conditioning(torch.tensor(audio)[None].to(device)).cpu()
        # Codec hop after 4x mel downsampling is .04 s. Use actual padded grid.
        conditioning_duration=length/16000
        rms=librosa.feature.rms(y=audio,frame_length=1024,hop_length=160)[0]
        centroid=librosa.feature.spectral_centroid(y=audio,sr=16000,n_fft=1024,hop_length=160)[0]
        times=(np.arange(len(rms))*160/16000).tolist();times[-1]=conditioning_duration
        controls=dict(row.get('controls',{}))
        controls.update({'rms_envelope':{'times_s':times,'values':rms.tolist()},'centroid_hz':{'times_s':times,'values':centroid.tolist()}})
        request={'family':row['family'],'controls':controls}
        cache=output/f'{i:06d}.pt';torch.save({'latent':latent,'condition':condition,'request':request,'conditioning_duration_s':conditioning_duration,'requested_duration_s':duration},cache)
        items.append({**row,'path':str(path),'sha256':digest,'cache_path':str(cache),'conditioning_duration_s':conditioning_duration})
    index={'status':'cache_prepared','records':items,'preprocessing':'mono16k, crop, mean-center, peak .5, zero-pad to codec grid; controls measured afterward','source_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    (output/'index.json').write_text(json.dumps(index,indent=2));return index
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args();print(json.dumps(prepare(ROOT,a.manifest,a.device),indent=2))
