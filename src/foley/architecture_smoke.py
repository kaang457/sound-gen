"""Actual frozen AudioLDM integration check, not a trained Foley-control result."""
import json
import time
from pathlib import Path
import sys
import numpy as np
import librosa
import soundfile as sf
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.foley.controls import ControlSpec,ControlBank
from src.foley.adapter import EpsilonAdapter,training_loss
from src.foley.backbone import load_backbone
from src.audio.footstep_sample import prepare_footstep

def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    model,config=load_backbone(ROOT)
    from audioldm.audio import TacotronSTFT
    from audioldm.audio.tools import get_mel_from_wav
    if '--skip-inference' in sys.argv:
        path=ROOT/'data/raw/footsteps_freefire66_5s_9s_22050.wav'
        source=json.loads((ROOT/'experiments/architecture/footstep_prototype.json').read_text())['source']
    else:path,source=prepare_footstep(ROOT)
    y,sr=sf.read(path,dtype='float32');y=librosa.resample(y,orig_sr=sr,target_sr=16000)
    y=y-y.mean();y=.5*y/(max(abs(y))+1e-8);y=np.pad(y,(0,81920-len(y))).astype(np.float32)
    # Upstream uses positional librosa APIs that became keyword-only.
    import audioldm.audio.stft as legacy_stft
    legacy_stft.pad_center=lambda data,size:librosa.util.pad_center(data,size=size)
    legacy_stft.librosa_mel_fn=lambda sr,n_fft,n_mels,fmin,fmax:librosa.filters.mel(sr=sr,n_fft=n_fft,n_mels=n_mels,fmin=fmin,fmax=fmax)
    stft=TacotronSTFT(1024,160,1024,64,16000,0,8000)
    mel,_,_=get_mel_from_wav(y,stft);mel=mel.T[:512]
    with torch.no_grad():
        clean=model.get_first_stage_encoding(model.encode_first_stage(torch.tensor(mel)[None,None]))
        condition=model.get_learned_conditioning(torch.tensor(y)[None])
    # Store a semantic prototype from one real exemplar; not a learned class model.
    prototype={'event_id':'footstep','family':'footstep','condition':condition.cpu().tolist(),'support_count':1,'source':source,'conditioning':'single-exemplar CLAP audio prototype; no text label encoding','model_revision':'4054cb418ba3d947d94f6ad302f1d6a320e2313c'}
    destination=ROOT/'experiments/architecture';destination.mkdir(parents=True,exist_ok=True)
    if '--skip-inference' not in sys.argv:(destination/'footstep_prototype.json').write_text(json.dumps(prototype,indent=2))
    data=json.loads((ROOT/'configs/foley_controls.json').read_text())
    bank=ControlBank([ControlSpec(**x) for x in data['controls']],data['width']);adapter=EpsilonAdapter(bank)
    rms=librosa.feature.rms(y=y,frame_length=1024,hop_length=160)[0]
    centroid=librosa.feature.spectral_centroid(y=y,sr=16000,n_fft=1024,hop_length=160)[0]
    times=np.linspace(0,5,len(rms)).tolist()
    request={'family':'footstep','controls':{'rms_envelope':{'times_s':times,'values':rms.tolist()},'centroid_hz':{'times_s':times,'values':centroid.tolist()}}}
    t=torch.tensor([500]);noise=torch.randn_like(clean);noisy=model.q_sample(clean,t,noise=noise)
    with torch.no_grad():base=model.apply_model(noisy,t,condition);zero=adapter(noisy,t,[request],5,condition)
    assert torch.count_nonzero(zero)==0
    opt=torch.optim.Adam(adapter.parameters(),lr=1e-4);steps=[];start=time.monotonic()
    for index in range(2):
        opt.zero_grad();loss,metrics=training_loss(model,adapter,clean,condition,[request],5,t=t,noise=noise)
        loss.backward();gradients=[p.grad for p in adapter.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g)>0 for g in gradients)
        opt.step();steps.append({'step':index,'loss':float(loss.detach()),**metrics})
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    with torch.no_grad():absent=adapter(noisy,t,[{'family':'footstep','controls':{}}],5,condition)
    assert torch.count_nonzero(absent)==0
    report={'status':'integration_smoke_passed','device':'cpu','torch':torch.__version__,'latent_shape':list(clean.shape),'condition_shape':list(condition.shape),'trainable_adapter_parameters':sum(p.numel() for p in adapter.parameters()),'frozen_backbone':True,'zero_initialization_preserves_epsilon':True,'absent_controls_zero_after_optimizer':True,'optimizer_steps':steps,'two_step_seconds':time.monotonic()-start,'source':source,'scope':'Two optimization steps on one real encoded recording. Numerical integration/gradient check only; no trained control accuracy, quality, generalization or new-parameter retention demonstrated.'}
    (destination/'audioldm_adapter_smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    if '--skip-inference' in sys.argv:return
    # Inference with cached audio condition, no reference WAV or user text at runtime.
    model.latent_t_size=128;torch.manual_seed(42)
    with torch.no_grad():
        unconditional=model.cond_stage_model.get_unconditional_condition(1)
        latent,_=model.sample_log(cond=condition,batch_size=1,ddim=True,ddim_steps=50,unconditional_guidance_scale=2.5,unconditional_conditioning=unconditional,eta=0.)
        audio=model.mel_spectrogram_to_waveform(model.decode_first_stage(latent))
    output=np.asarray(audio).reshape(-1).astype(np.float32)[:80000]
    folder=ROOT/'results/audio/foley';folder.mkdir(parents=True,exist_ok=True);sf.write(folder/'footstep_prototype_baseline.wav',output,16000,subtype='FLOAT')
    report={'status':'baseline_inference_completed','event_id':'footstep','user_text_prompt':None,'runtime_reference_wav_required':False,'positive_condition':'cached single-exemplar audio embedding','steps':50,'seed':42,'duration_s':len(output)/16000,'output_rms':float(np.sqrt(np.mean(output.astype(np.float64)**2))),'output_peak':float(max(abs(output))),'numeric_controls_applied':False,'quality_evaluation':'not measured','scope':'Pretrained backbone inference; adapter not trained for usable controls.'}
    (destination/'footstep_prototype_baseline.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
