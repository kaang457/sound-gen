"""Minimal class + RMS inference adapter for the pinned official T-FOLEY model.

Uses the upstream model/sampler/sde with one device-placement correction in
RFF embedding, and soundfile I/O instead of the CUDA-only CLI. No text prompt.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import ellip, filtfilt, firwin, lfilter

UPSTREAM = 'https://github.com/YoonjinXD/T-FOLEY.git'
REVISION = '7a0fb41c193625b7edbed86e823e4b91b55c886c'
CHECKPOINT_URL = 'https://zenodo.org/records/10826692/files/pretrained.zip?download=1'
ZIP_MD5 = '3eec767874a780a48e7eea2a415d1f94'
LABELS = ['DogBark','Footstep','GunShot','Keyboard','MovingMotorVehicle','Rain','Sneeze_Cough']


def md5_file(path):
    h=hashlib.md5()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def prepare(root):
    external=Path(root)/'external'/'T-FOLEY'
    if not external.exists():
        external.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone',UPSTREAM,str(external)],check=True)
    current=subprocess.check_output(['git','-C',str(external),'rev-parse','HEAD'],text=True).strip()
    if current!=REVISION:
        subprocess.run(['git','-C',str(external),'checkout','--detach',REVISION],check=True)
    weights=Path(root)/'models'/'tfoley'
    weights.mkdir(parents=True,exist_ok=True)
    archive=weights/'pretrained.zip'
    if not archive.exists():
        tmp=archive.with_suffix('.download')
        try:
            urllib.request.urlretrieve(CHECKPOINT_URL,tmp)
            if md5_file(tmp)!=ZIP_MD5:raise ValueError('Checkpoint checksum mismatch.')
            tmp.replace(archive)
        finally:tmp.unlink(missing_ok=True)
    if md5_file(archive)!=ZIP_MD5:raise ValueError('Checkpoint checksum mismatch.')
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            dest=(weights/entry.filename).resolve()
            if not dest.is_relative_to(weights.resolve()):raise ValueError('Invalid archive path.')
        z.extractall(weights)
    return external,weights


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--upstream-dir',type=Path)
    p.add_argument('--weights-dir',type=Path)
    p.add_argument('--class-name',choices=LABELS,default='Footstep')
    p.add_argument('--target-audio',type=Path)
    p.add_argument('--rms-json',type=Path, help='Direct 690-frame nonnegative RMS condition; mutually exclusive with target audio')
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output-name',default='smoke_test')
    args=p.parse_args()
    if args.rms_json and args.target_audio:raise ValueError('Choose rms-json or target-audio')
    if args.steps<2:raise ValueError('steps must be >= 2')
    if Path(args.output_name).name!=args.output_name:raise ValueError('output-name must be a filename stem')
    root=Path(__file__).resolve().parents[2]
    if args.prepare:
        upstream,weights=prepare(root)
    else:
        upstream=args.upstream_dir or root/'external'/'T-FOLEY'
        weights=args.weights_dir or root/'models'/'tfoley'
    actual_revision=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if actual_revision!=REVISION:raise ValueError('Upstream revision mismatch.')
    params_paths=list(weights.rglob('params.json'))
    model_paths=list(weights.rglob('block-49_epoch-500.pt'))
    if len(params_paths)!=1 or len(model_paths)!=1:raise FileNotFoundError('Expected exactly one params.json and block-49_epoch-500.pt; run --prepare.')
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA not available')
    sys.path.insert(0,str(upstream.resolve()))
    from model import UNet, RFF_MLP_Block
    from sampler import SDESampling_batch
    from sde import VpSdeCos
    # Upstream hardcodes CUDA here. Preserve the formula, use the input device.
    def portable_rff_embedding(self, sigma):
        table = 2 * np.pi * sigma * self.RFF_freq.to(sigma.device)
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    RFF_MLP_Block._build_RFF_embedding = portable_rff_embedding
    params=json.loads(params_paths[0].read_text())
    sr=int(params['sample_rate']);length=4*sr
    if params['event_type']!='rms':raise ValueError('This adapter expects RMS conditioning.')
    model=UNet(len(LABELS),params).to(device)
    checkpoint=torch.load(model_paths[0],map_location=device,weights_only=True)
    keys=list(checkpoint['model'].keys())
    if len(keys)!=len(checkpoint['ema_weights']):raise ValueError('EMA tensor count mismatch.')
    model.load_state_dict(dict(zip(keys,checkpoint['ema_weights'])),strict=True)
    del checkpoint
    model.eval()
    if args.target_audio:
        target,source_sr=sf.read(args.target_audio,dtype='float32',always_2d=True)
        target=target.mean(axis=1)
        if source_sr!=sr:target=librosa.resample(target,orig_sr=source_sr,target_sr=sr)
        target=np.pad(target[:length],(0,max(0,length-len(target))))
        source='user WAV (mono, resampled if necessary, crop/pad to four seconds)'
    else:
        t=np.arange(length)/sr
        target=np.zeros(length,dtype=np.float32)
        for onset in [0.6,1.8,3.0]:
            u=t-onset;m=u>=0
            target[m]+=0.7*np.exp(-u[m]/0.1)*np.sin(2*np.pi*220*u[m])
        source='synthetic three-pulse reference; not real footsteps'
    rms=librosa.feature.rms(y=target,frame_length=512,hop_length=128)[0]
    b,a=ellip(4,0.01,120,0.125)
    events=filtfilt(b,a,rms,method='gust').astype(np.float32)
    if len(events)!=params['event_dims']['rms']:raise ValueError('Condition length mismatch.')
    if args.rms_json:
        specification=json.loads(args.rms_json.read_text(encoding='utf-8'))
        events=np.asarray(specification['rms'],dtype=np.float32)
        if events.shape!=(params['event_dims']['rms'],) or not np.isfinite(events).all() or np.any(events<0):
            raise ValueError('rms-json requires exactly 690 finite nonnegative values')
        source='direct numeric RMS condition; reference waveform is not used'
    cond=torch.from_numpy(events.copy()).unsqueeze(0).to(device)
    sampler=SDESampling_batch(model,VpSdeCos(),batch_size=1,device=device)
    noise=torch.randn(1,length,device=device)
    classes=torch.tensor([LABELS.index(args.class_name)],device=device)
    start=time.monotonic()
    print(f'Inference: device={device}, steps={args.steps}, class={args.class_name}',flush=True)
    with torch.inference_mode():generated=sampler.predict(noise,args.steps,classes,cond,cond_scale=3)
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    samples=generated[0].cpu().numpy()
    samples=lfilter(firwin(101,cutoff=20,fs=sr,pass_zero='highpass'),[1,0],samples)
    if not np.isfinite(samples).all():raise ValueError('Nonfinite generated audio')
    output=root/'results'/'audio'/'tfoley';output.mkdir(parents=True,exist_ok=True)
    sf.write(output/(args.output_name+'.wav'),samples,sr,subtype='FLOAT')
    if not args.rms_json:sf.write(output/(args.output_name+'_reference.wav'),target,sr,subtype='FLOAT')
    (output/(args.output_name+'_condition.json')).write_text(json.dumps({'rms':events.tolist(),'sample_rate':sr,'hop_length':128}),encoding='utf-8')
    report={'status':'inference_completed','time_utc':datetime.now(timezone.utc).isoformat(),
            'upstream_revision':actual_revision,'checkpoint_sha256':hashlib.sha256(model_paths[0].read_bytes()).hexdigest(),
            'device':str(device),'torch':torch.__version__,'seed':args.seed,'steps':args.steps,
            'class_name':args.class_name,'sample_rate':sr,'duration_s':4,'cond_scale':3,
            'reference':source,'reference_sha256':None if args.rms_json else hashlib.sha256(target.astype('<f4').tobytes()).hexdigest(),
            'condition_sha256':hashlib.sha256(events.astype('<f4').tobytes()).hexdigest(),
            'conditioning': 'direct_rms' if args.rms_json else 'reference_rms',
            'inference_seconds':elapsed,'parameter_count':sum(p.numel() for p in model.parameters()),
            'output_peak':float(np.max(np.abs(samples))),'output_rms':float(np.sqrt(np.mean(samples**2))),
            'output_wav_sha256':hashlib.sha256((output/(args.output_name+'.wav')).read_bytes()).hexdigest(),
            'quality_evaluation':'not measured','paper_reproduction':False,
            'adapter_changes':['RFF frequencies use sigma.device instead of hardcoded CUDA', 'soundfile I/O; portable checkpoint loading'],
            'packages':{n:importlib.metadata.version(n) for n in ['numpy','librosa','soundfile','einops']}}
    dest=root/'experiments'/'baselines'/'tfoley';dest.mkdir(parents=True,exist_ok=True)
    (dest/(args.output_name+'_run.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()

