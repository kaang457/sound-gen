"""Reuse audited full AudioLDM weights; positive conditions use audio, not text."""
from pathlib import Path
import sys
import subprocess
import torch

def load_backbone(root,device='cpu',seed=42):
    if device not in ('cpu','cuda'):raise ValueError('device must be cpu or cuda')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in model environment')
    root=Path(root).resolve();sys.path.insert(0,str(root))
    from src.baselines.audioldm_reference_inference import REVISION,MODEL_NAME,CHECKPOINT_SIZE,CHECKPOINT_MD5,digest_file,prepare_tokenizer
    upstream=root/'external/AudioLDM';checkpoint=root/'models/audioldm/audioldm-m-full.ckpt'
    if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=REVISION:raise ValueError('Upstream revision mismatch')
    if checkpoint.stat().st_size!=CHECKPOINT_SIZE or digest_file(checkpoint,'md5')!=CHECKPOINT_MD5:raise ValueError('Checkpoint mismatch; run notebook08 preparation')
    sys.path.insert(0,str(upstream))
    from audioldm import LatentDiffusion,seed_everything
    from audioldm.pipeline import set_cond_audio
    from audioldm.utils import default_audioldm_config
    from transformers import RobertaConfig,RobertaModel,RobertaTokenizer
    folder=prepare_tokenizer(root)
    # Restore class descriptors after construction; do not leave global patches.
    original_tokenizer=RobertaTokenizer.from_pretrained
    tokenizer_descriptor=RobertaTokenizer.__dict__['from_pretrained'] if 'from_pretrained' in RobertaTokenizer.__dict__ else None
    model_descriptor=RobertaModel.__dict__.get('from_pretrained')
    def local_tokenizer(cls,identifier,*args,**kwargs):
        if identifier!='roberta-base':raise ValueError('Unexpected tokenizer')
        return original_tokenizer(str(folder),local_files_only=True)
    def local_model(cls,identifier,*args,**kwargs):
        if identifier!='roberta-base':raise ValueError('Unexpected model')
        return cls(RobertaConfig.from_pretrained(str(folder),local_files_only=True))
    RobertaTokenizer.from_pretrained=classmethod(local_tokenizer);RobertaModel.from_pretrained=classmethod(local_model)
    try:
        seed_everything(seed);config=default_audioldm_config(MODEL_NAME)
        config['model']['params']['device']=device;config['model']['params']['cond_stage_key']='waveform'
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        model=LatentDiffusion(**config['model']['params']);keys=model.load_state_dict(weights['state_dict'],strict=False)
        if keys.missing_keys or set(keys.unexpected_keys)-{'cond_stage_model.model.text_branch.embeddings.position_ids'}:raise ValueError('Checkpoint keys mismatch')
        del weights
    finally:
        if tokenizer_descriptor is None:del RobertaTokenizer.from_pretrained
        else:RobertaTokenizer.from_pretrained=tokenizer_descriptor
        if model_descriptor is None:del RobertaModel.from_pretrained
        else:RobertaModel.from_pretrained=model_descriptor
    model=set_cond_audio(model.eval().to(device))
    # Install inference EMA weights before freezing. Legacy ema_scope assumes
    # trainable UNet parameters and asserts if entered after requires_grad=False.
    if model.use_ema:model.model_ema.copy_to(model.model)
    model.requires_grad_(False)
    model.cond_stage_model.unconditional_prob=0
    return model,config
