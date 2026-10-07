"""Small conditional spectral VAE, trained on one licensed footstep excerpt.

A limited engineering experiment, not F-RAVE reproduction or general Foley model.
RMS and centroid enter the neural decoder. No target gain/filter correction.
Phase is sampled by a fixed renderer, so this is neural spectral synthesis.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.audio.footstep_sample import prepare_footstep
N=512
SR=22050

class SpectralCVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(259,64),nn.SiLU(),nn.Linear(64,4))
        self.decoder=nn.Sequential(nn.Linear(4,32),nn.SiLU(),nn.Linear(32,64),nn.SiLU(),nn.Linear(64,257),nn.Softplus())
    def encode(self,x,c):
        mu,logvar=self.encoder(torch.cat([torch.log1p(x),c],1)).chunk(2,1)
        return mu,logvar.clamp(-8,8)
    def decode(self,c,z): return self.decoder(torch.cat([c,z],1))

def descriptors(m):
    weights=torch.ones(257,device=m.device);weights[1:-1]=2
    rms=torch.sqrt(torch.sum(m*m*weights,dim=1)+1e-12)/N
    f=torch.linspace(0,SR/2,257,device=m.device)
    centroid=torch.sum(m*f,dim=1)/(m.sum(1)+1e-12)
    return torch.stack([rms,centroid],1)

def pack(c,mean,std): return (torch.stack([torch.log(c[:,0].clamp_min(1e-7)),c[:,1]/(SR/2)],1)-mean)/std

def train(root=ROOT,steps=3000):
    root=Path(root);torch.set_num_threads(2);torch.manual_seed(17)
    path,metadata=prepare_footstep(root);y,sr=sf.read(path)
    if sr!=SR:raise ValueError('Expected 22050 Hz')
    frames=y[:len(y)//N*N].reshape(-1,N)
    spectra=torch.tensor(abs(np.fft.rfft(frames)),dtype=torch.float32)
    c=descriptors(spectra)
    # Exclude almost silent blocks; split contiguous time ranges before fitting.
    indices=np.arange(len(frames));cut1=int(.6*len(frames));cut2=int(.8*len(frames))
    masks=[(indices<cut1),(indices>=cut1)&(indices<cut2),(indices>=cut2)]
    masks=[torch.tensor(m)&(c[:,0]>1e-4) for m in masks]
    train_x=spectra[masks[0]];train_c=c[masks[0]]
    scale=float(train_c[:,0].median());train_x=train_x/scale
    raw=torch.stack([torch.log(train_c[:,0]),train_c[:,1]/(SR/2)],1)
    mean=raw.mean(0);std=raw.std(0).clamp_min(.05)
    model=SpectralCVAE();opt=torch.optim.Adam(model.parameters(),lr=.002)
    history=[]
    for step in range(steps):
        idx=torch.randint(len(train_x),(64,));x=train_x[idx];control=train_c[idx];cond=pack(control,mean,std)
        mu,lv=model.encode(x,cond);z=mu+torch.randn_like(mu)*torch.exp(.5*lv)
        output=model.decode(cond,z);features=descriptors(output*scale)
        reconstruction=(torch.log1p(output)-torch.log1p(x)).square().mean()
        control_loss=(torch.log(features[:,0]/control[:,0])).square().mean()+((features[:,1]-control[:,1])/1000).square().mean()
        kl=-.5*(1+lv-mu.square()-lv.exp()).mean()
        prior_features=descriptors(model.decode(cond,torch.randn_like(mu))*scale)
        prior_control=(torch.log(prior_features[:,0]/control[:,0])).square().mean()+((prior_features[:,1]-control[:,1])/1000).square().mean()
        loss=reconstruction+5*control_loss+10*prior_control+.1*kl
        opt.zero_grad();loss.backward();opt.step()
        if step%500==0:history.append({'step':step,'loss':float(loss.detach()),'control_loss':float(control_loss.detach())})
    model.eval()
    artifact={'format':'spectral-cvae-v1','sample_rate':SR,'fft_size':N,'scale':scale,'mean':mean.tolist(),'std':std.tolist(),'state_dict':{k:v.detach().tolist() for k,v in model.state_dict().items()},'training_steps':steps,'seed':17,'objective':'log1p magnitude reconstruction + 5 posterior control + 10 prior control + .1 KL','source':metadata,'training_control_min':train_c.min(0).values.tolist(),'training_control_max':train_c.max(0).values.tolist()}
    folder=root/'experiments/parametric/spectral_cvae';folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/'weights.json';checkpoint.write_text(json.dumps(artifact),encoding='utf-8')
    errors={}
    with torch.no_grad():
        for label,mask in zip(['train','validation','test'],masks):
            control=c[mask];cond=pack(control,mean,std)
            # Prior sampling, not posterior reconstruction; held-out controls used.
            out=model.decode(cond,torch.randn(len(control),2))*scale;features=descriptors(out)
            errors[label]={'frames':len(control),'rms_relative_mae':float(((features[:,0]-control[:,0]).abs()/control[:,0]).mean()),'centroid_mae_hz':float((features[:,1]-control[:,1]).abs().mean())}
    report={'status':'training_completed','checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'steps':steps,'objective':'log1p magnitude reconstruction + 5 posterior control + 10 prior control + .1 KL','device':'cpu','threads':2,'timing':'Not measured','parameter_count':sum(p.numel() for p in model.parameters()),'packages':{'torch':torch.__version__,'numpy':np.__version__,'soundfile':sf.__version__},'history':history,'prior_sampling_metrics':errors,'split':'Contiguous nonoverlapping 512-sample blocks: first 60% train, next 20% validation, last 20% test. Statistics fit on train only. Initial exploratory metrics motivated a prior-control loss change; these are development diagnostics, not an untouched test benchmark. One recording, no recording-level generalization.','scope':'Real-recording magnitude-spectrum CVAE. Fixed random-phase renderer. No learned temporal structure, category guarantee, pitch control or F-RAVE reproduction.'}
    (folder/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2));return report

def generate(root=ROOT,rms=.02,centroid_hz=3500,duration_s=2,seed=42,name='cvae_sample'):
    root=Path(root)
    if not name or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in name):raise ValueError('Invalid output stem')
    if not np.isfinite([rms,centroid_hz,duration_s]).all() or rms<=0 or not 0<centroid_hz<SR/2 or not 0<duration_s<=10:raise ValueError('Invalid controls')
    a=json.loads((root/'experiments/parametric/spectral_cvae/weights.json').read_text())
    bounds=np.array([a['training_control_min'],a['training_control_max']])
    if np.any(np.array([rms,centroid_hz])<bounds[0]) or np.any(np.array([rms,centroid_hz])>bounds[1]):raise ValueError(f'Controls outside marginal training range: {bounds.tolist()}')
    torch.set_num_threads(2);torch.manual_seed(seed);model=SpectralCVAE();model.load_state_dict({k:torch.tensor(v) for k,v in a['state_dict'].items()});model.eval()
    count=max(1,round(duration_s*SR/N));control=torch.tensor([[rms,centroid_hz]]*count)
    with torch.no_grad():m=model.decode(pack(control,torch.tensor(a['mean']),torch.tensor(a['std'])),torch.randn(count,2)).numpy()*a['scale']
    rng=np.random.default_rng(seed);phase=rng.uniform(-np.pi,np.pi,m.shape);phase[:,[0,-1]]=0
    blocks=np.fft.irfft(m*np.exp(1j*phase),n=N);y=blocks.reshape(-1)
    folder=root/'results/audio/parametric';folder.mkdir(parents=True,exist_ok=True);sf.write(folder/(name+'.wav'),y,SR,subtype='FLOAT')
    # Measure rendered blocks, not a feature-loss tensor; no gain normalization.
    actual=descriptors(torch.tensor(abs(np.fft.rfft(blocks)),dtype=torch.float32)).numpy()
    result={'name':name,'seed':seed,'target_rms':rms,'target_centroid_hz':centroid_hz,'actual_mean_block_rms':float(actual[:,0].mean()),'actual_mean_block_centroid_hz':float(actual[:,1].mean()),'duration_s':len(y)/SR,'output_peak':float(abs(y).max()),'fraction_above_1':float(np.mean(abs(y)>1)),'wav_sha256':hashlib.sha256((folder/(name+'.wav')).read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256((root/'experiments/parametric/spectral_cvae/weights.json').read_bytes()).hexdigest(),'scope':'Block RMS and rectangular FFT magnitude centroid; not interchangeable with Hann/STFT whole-recording metrics. Controls within marginal ranges may lie outside joint training support.'}
    return y,SR,result

def sweep(root=ROOT):
    root=Path(root);a=json.loads((root/'experiments/parametric/spectral_cvae/weights.json').read_text());lo=np.array(a['training_control_min']);hi=np.array(a['training_control_max'])
    # Interior marginal range; report joint support limitation explicitly.
    values=lo+(hi-lo)*np.array([[.25,.5],[.75,.5],[.5,.25],[.5,.75]])
    rows=[]
    for seed in [42,43,44]:
        for i,(rms,centroid) in enumerate(values):
            _,_,row=generate(root,float(rms),float(centroid),seed=seed,name=f'cvae_{i}_seed{seed}');rows.append(row)
    rms_tests=[];centroid_tests=[]
    for seed in [42,43,44]:
        r=[x for x in rows if x['seed']==seed]
        rms_tests.append({'seed':seed,'rms_monotonic':r[1]['actual_mean_block_rms']>r[0]['actual_mean_block_rms'],'centroid_drift_hz':abs(r[1]['actual_mean_block_centroid_hz']-r[0]['actual_mean_block_centroid_hz'])})
        centroid_tests.append({'seed':seed,'centroid_monotonic':r[3]['actual_mean_block_centroid_hz']>r[2]['actual_mean_block_centroid_hz'],'rms_relative_drift':abs(r[3]['actual_mean_block_rms']-r[2]['actual_mean_block_rms'])/r[2]['target_rms']})
    report={'results':rows,'rms_sweep':rms_tests,'centroid_sweep':centroid_tests,'scope':'12 clips, 3 seeds. Small single-recording proof of learned two-control spectral synthesis; no Foley quality or broad generalization claim.'}
    (root/'experiments/parametric/spectral_cvae/sweep.json').write_text(json.dumps(report,indent=2))
    import csv
    import matplotlib.pyplot as plt
    folder=root/'results/tables/parametric';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'spectral_cvae_sweep.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,key,actual,label in [(axes[0],'target_rms','actual_mean_block_rms','RMS'),(axes[1],'target_centroid_hz','actual_mean_block_centroid_hz','Centroid (Hz)')]:
        for seed in [42,43,44]:
            values=[r for r in rows if r['seed']==seed];ax.scatter([r[key] for r in values],[r[actual] for r in values],label=f'Seed {seed}')
        lo=min(r[key] for r in rows);hi=max(r[key] for r in rows);ax.plot([lo,hi],[lo,hi],'k--',label='Target = output');ax.set(xlabel=f'Target {label}',ylabel=f'Mean rendered block {label}');ax.legend()
    folder=root/'results/figures/parametric';folder.mkdir(parents=True,exist_ok=True);fig.savefig(folder/'spectral_cvae_sweep.svg');plt.close(fig)
    print(json.dumps(report,indent=2));return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',action='store_true');p.add_argument('--steps',type=int,default=3000);a=p.parse_args()
    if a.train:train(steps=a.steps)
    sweep()
