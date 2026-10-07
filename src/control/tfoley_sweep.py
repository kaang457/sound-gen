"""Paired condition/seed sweep. Metrics concern energy, not sound identity."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import ellip, filtfilt, find_peaks
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.control.tfoley import generate, pulse_curve

def run(root=ROOT,steps=50,device='auto',prepare=False):
    root=Path(root); rows=[]; curves=[]
    cases=[('low',.015,(.6,1.8,3.)),('high',.045,(.6,1.8,3.)),('shift',.03,(1.,2.2,3.4))]
    for seed in [42,43]:
        for label,level,onsets in cases:
            name=f'control_{label}_seed{seed}';target=pulse_curve(level,onsets)
            path=generate(root,target,name,seed,steps,device=device,prepare=prepare)
            y,sr=sf.read(path);b,a=ellip(4,.01,120,.125)
            actual=filtfilt(b,a,librosa.feature.rms(y=y.astype(np.float32),frame_length=512,hop_length=128)[0],method='gust')
            peaks=find_peaks(actual,distance=round(.25*sr/128),prominence=max(1e-8,.15*float(actual.max())))[0]*128/sr
            distances=abs(np.asarray(onsets)[:,None]-peaks[None,:]); matched=[]
            if len(peaks):
                i,j=linear_sum_assignment(distances)
                matched=[float(distances[x,z]) for x,z in zip(i,j) if distances[x,z]<=.15]
            rows.append(dict(name=name,seed=seed,target_peak_rms=level,mae=float(np.mean(abs(actual-target))),normalized_mae=float(np.mean(abs(actual-target))/level),pearson=float(np.corrcoef(target,actual)[0,1]) if np.std(actual)>0 else None,output_rms=float(np.sqrt(np.mean(y*y))),output_peak=float(abs(y).max()),fraction_above_1=float(np.mean(abs(y)>1)),detected_rms_peaks=len(peaks),matched_within_150ms=len(matched),missing_target_peaks=len(onsets)-len(matched),extra_detected_peaks=len(peaks)-len(matched),mean_matched_offset_s=float(np.mean(matched)) if matched else None))
            curves.append((name,target,actual))
    folder=root/'experiments/parametric';folder.mkdir(parents=True,exist_ok=True)
    monotonic={str(seed):next(r['output_rms'] for r in rows if r['name']==f'control_high_seed{seed}')>next(r['output_rms'] for r in rows if r['name']==f'control_low_seed{seed}') for seed in [42,43]}
    report={'status':'completed','steps':steps,'seeds':[42,43],'results':rows,'high_over_low_rms':monotonic,'scope':'Six samples, two seeds; RMS peaks not semantic events. Hungarian one-to-one assignment, 150ms acceptance. No perceptual quality or paper benchmark claim.'}
    (folder/'tfoley_sweep.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    table=root/'results/tables/parametric';table.mkdir(parents=True,exist_ok=True)
    with (table/'tfoley_sweep.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(3,2,figsize=(11,8),layout='constrained')
    for ax,(name,target,actual) in zip(axes.T.flat,curves):
        t=np.arange(690)*128/22050;ax.plot(t,target,label='Target');ax.plot(t,actual,label='Output');ax.set(title=name,xlabel='s',ylabel='RMS');ax.legend()
    dest=root/'results/figures/parametric';dest.mkdir(parents=True,exist_ok=True);fig.savefig(dest/'tfoley_sweep.svg');plt.close(fig)
    print(json.dumps(report,indent=2))
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=50);p.add_argument('--device',default='auto');p.add_argument('--prepare',action='store_true');a=p.parse_args();run(steps=a.steps,device=a.device,prepare=a.prepare)
