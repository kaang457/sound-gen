"""Pinned official AudioLDM m-full, conditioned by reference audio, one candidate."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request
import numpy as np
import librosa
import soundfile as sf

REVISION = '4054cb418ba3d947d94f6ad302f1d6a320e2313c'
UPSTREAM = 'https://github.com/haoheliu/AudioLDM.git'
URL = 'https://zenodo.org/records/7813012/files/audioldm-m-full.ckpt?download=1'
CHECKPOINT_SIZE = 4571683377
CHECKPOINT_MD5 = '46bad9f176651404b3cf1484942749b9'
MODEL_NAME = 'audioldm-m-full'
TOKENIZER_REVISION = 'e2da8e2f811d1448a5b465c236feacd80ffbac7b'
TOKENIZER_SHA256 = {
    'config.json': 'ef0185e2aae6e06c5f105a285006952c340e20c7dbf43c86ec82601b13fc45e9',
    'tokenizer_config.json': '994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce',
    'vocab.json': '9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655',
    'merges.txt': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
}


def prepare_tokenizer(root):
    folder = root / 'models/audioldm/roberta-base'
    folder.mkdir(parents=True, exist_ok=True)
    for name in ['config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']:
        path = folder / name
        if not path.exists():
            urllib.request.urlretrieve(f'https://huggingface.co/roberta-base/resolve/{TOKENIZER_REVISION}/{name}', path)
        if digest_file(path) != TOKENIZER_SHA256[name]:
            raise ValueError(f'Tokenizer/config checksum mismatch: {name}')
    return folder


def digest_file(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        while block := f.read(1024 * 1024): h.update(block)
    return h.hexdigest()


def prepare(root):
    upstream = root / 'external/AudioLDM'
    if not upstream.exists():
        upstream.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', UPSTREAM, str(upstream)], check=True)
        subprocess.run(['git', '-C', str(upstream), 'checkout', '--detach', REVISION], check=True)
    checkpoint = root / 'models/audioldm/audioldm-m-full.ckpt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        temp = checkpoint.with_suffix('.download')
        offset = temp.stat().st_size if temp.exists() else 0
        if offset > CHECKPOINT_SIZE: raise ValueError('Partial checkpoint exceeds expected size')
        if offset < CHECKPOINT_SIZE:
            req = urllib.request.Request(URL, headers={'Range': f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(req, timeout=120) as response:
                if offset and response.status == 206:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Invalid resume response')
                    mode = 'ab'
                else:
                    mode, offset = 'wb', 0
                next_notice = offset + 256 * 1024 * 1024
                with temp.open(mode) as f:
                    while block := response.read(1024 * 1024):
                        f.write(block); offset += len(block)
                        if offset >= next_notice:
                            print(f'Checkpoint: {offset / 1e9:.2f} / {CHECKPOINT_SIZE / 1e9:.2f} GB', flush=True)
                            next_notice += 256 * 1024 * 1024
        if temp.stat().st_size != CHECKPOINT_SIZE or digest_file(temp, 'md5') != CHECKPOINT_MD5:
            temp.unlink(missing_ok=True)
            raise ValueError('Checkpoint size/checksum mismatch')
        temp.replace(checkpoint)
    return upstream, checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--upstream-dir', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--duration', type=float, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--output-name', default='reference_run')
    args = p.parse_args()
    if args.steps < 2 or not 2.5 <= args.duration <= 10:
        raise ValueError('steps >= 2 and duration between 2.5 and 10 s required')
    if not args.output_name or Path(args.output_name).name != args.output_name:
        raise ValueError('output-name must be a filename stem')
    root = Path(__file__).resolve().parents[2]
    if args.prepare:
        upstream, checkpoint = prepare(root)
    else:
        upstream = args.upstream_dir or root / 'external/AudioLDM'
        checkpoint = args.checkpoint or root / 'models/audioldm/audioldm-m-full.ckpt'
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION: raise ValueError('Upstream revision mismatch')
    if checkpoint.stat().st_size != CHECKPOINT_SIZE or digest_file(checkpoint, 'md5') != CHECKPOINT_MD5:
        raise ValueError('Checkpoint checksum mismatch')
    import torch
    torch.set_num_threads(args.threads)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    if device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    sys.path.insert(0, str(upstream.resolve()))
    from audioldm import LatentDiffusion, seed_everything
    from audioldm.utils import default_audioldm_config
    from audioldm.pipeline import make_batch_for_text_to_audio, set_cond_audio, duration_to_latent_t_size
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
    tokenizer_folder = prepare_tokenizer(root)
    original_tokenizer_loader = RobertaTokenizer.from_pretrained
    def local_tokenizer(cls, identifier, *unused_args, **unused_kwargs):
        if identifier != 'roberta-base': raise ValueError('Unexpected tokenizer identifier')
        return original_tokenizer_loader(str(tokenizer_folder), local_files_only=True)
    def checkpoint_roberta(cls, identifier, *unused_args, **unused_kwargs):
        if identifier != 'roberta-base': raise ValueError('Unexpected text encoder identifier')
        return cls(RobertaConfig.from_pretrained(str(tokenizer_folder), local_files_only=True))
    # AudioLDM checkpoint already contains all 200 RoBERTa tensors. Avoid
    # downloading a second, temporary encoder that would be overwritten.
    RobertaTokenizer.from_pretrained = classmethod(local_tokenizer)
    RobertaModel.from_pretrained = classmethod(checkpoint_roberta)
    seed_everything(args.seed)
    config = default_audioldm_config(MODEL_NAME)
    config['model']['params']['device'] = device
    config['model']['params']['cond_stage_key'] = 'waveform'
    print(f'Loading {MODEL_NAME}; positive condition: audio; device: {device}', flush=True)
    # mmap avoids holding another full 4.57 GB checkpoint copy in RAM.
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    model = LatentDiffusion(**config['model']['params'])
    incompatible = model.load_state_dict(weights['state_dict'], strict=False)
    allowed_extra = {'cond_stage_model.model.text_branch.embeddings.position_ids'}
    if incompatible.missing_keys or set(incompatible.unexpected_keys) - allowed_extra:
        raise ValueError(f'Unexpected checkpoint mismatch: {incompatible}')
    del weights
    model.eval().to(device)
    model = set_cond_audio(model)
    # The legacy conditioner applies this drop even in eval mode.
    model.cond_stage_model.unconditional_prob = 0.0
    # Never run multi-candidate text scoring; guard even an accidental call.
    def forbidden_text_scoring(*unused_args, **unused_kwargs):
        raise RuntimeError('Text candidate scoring is disabled in this audio-reference experiment')
    model.cond_stage_model.cos_similarity = forbidden_text_scoring
    audio, original_sr = sf.read(args.reference, dtype='float32', always_2d=True)
    audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all(): raise ValueError('Invalid reference audio')
    reference = librosa.resample(audio, orig_sr=original_sr, target_sr=16000)
    # Match upstream read_wav_file: center and scale to peak .5, then crop/pad.
    centered = reference - np.mean(reference)
    if np.max(abs(centered)) < 1e-8: raise ValueError('Reference is silent or constant')
    centered = 0.5 * centered / (np.max(abs(centered)) + 1e-8)
    condition_length = int(args.duration * 102.4) * 160
    conditioning = np.pad(centered[:condition_length], (0, max(0, condition_length-len(centered))))
    conditioning = (0.5 * conditioning / np.max(abs(conditioning))).astype(np.float32)
    batch = make_batch_for_text_to_audio('', waveform=conditioning[None, :], batchsize=1)
    model.latent_t_size = duration_to_latent_t_size(args.duration)
    start = time.monotonic()
    with torch.no_grad():
        result = model.generate_sample([batch], unconditional_guidance_scale=2.5,
                                      ddim_steps=args.steps, n_candidate_gen_per_text=1,
                                      duration=args.duration)
    if device == 'cuda': torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    raw = np.asarray(result).reshape(-1)
    output = raw.astype(np.float32) / 32768 if raw.dtype == np.int16 else raw.astype(np.float32)
    decoded_length = len(output)
    target_length = round(args.duration * 16000)
    if len(output) < target_length or not np.isfinite(output).all(): raise ValueError('Invalid generated audio')
    output = output[:target_length]
    folder = root / 'results/audio/audioldm'; folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / (args.output_name+'.wav')
    sf.write(out_path, output, 16000, subtype='FLOAT')
    sf.write(folder / (args.output_name+'_reference.wav'), reference, 16000, subtype='FLOAT')
    sf.write(folder / (args.output_name+'_conditioning.wav'), conditioning, 16000, subtype='FLOAT')
    report = {'status':'inference_completed', 'time_utc':datetime.now(timezone.utc).isoformat(),
              'python':platform.python_version(), 'platform':platform.system(), 'device':device,
              'model_name':MODEL_NAME, 'upstream_revision':revision,
              'checkpoint_md5':CHECKPOINT_MD5, 'checkpoint_sha256':digest_file(checkpoint),
              'reference_wav_sha256':digest_file(args.reference),
              'positive_condition':'reference waveform -> checkpoint CLAP audio embedding',
              'user_text_prompt':None, 'empty_text_unconditional_branch':True,
              'candidate_count':1, 'text_candidate_scoring':False,
              'positive_condition_dropout_probability':0.0,
              'tokenizer_revision':TOKENIZER_REVISION,
              'tokenizer_config_sha256':TOKENIZER_SHA256,
              'seed':args.seed, 'steps':args.steps, 'guidance_scale':2.5,
              'requested_duration_s':args.duration, 'output_duration_s':len(output)/16000,
              'source_duration_s':len(audio)/original_sr, 'conditioning_duration_s':len(conditioning)/16000,
              'reference_preprocessing':'mono; resample 16k; mean-center; peak .5; crop/zero-pad; peak .5',
              'raw_vocoder_dtype':str(raw.dtype), 'decoded_samples_before_crop':decoded_length,
              'pcm16_conversion_applied':bool(raw.dtype == np.int16),
              'output_rms':float(np.sqrt(np.mean(output.astype(np.float64)**2))),
              'output_peak':float(np.max(abs(output))), 'sampling_seconds':elapsed,
              'timing_scope':'generate_sample: CLAP/latent encoding, DDIM, decoder/vocoder; excludes model load and I/O',
              'output_wav_sha256':digest_file(out_path),
              'quality_evaluation':'not measured', 'paper_reproduction':False,
              'adapter_changes':['soundfile/librosa reference I/O instead of torchaudio.load',
                                 'safe weights_only/mmap checkpoint load with key validation',
                                 'construct RoBERTa from local config; full text weights come from AudioLDM checkpoint',
                                 'zero positive-condition dropout; single candidate and forbidden text scoring',
                                 'PCM16 unit handling if present; crop decoded tail'],
              'packages':{n:importlib.metadata.version(n) for n in ['torch','torchaudio','torchvision','numpy','librosa','transformers','torchlibrosa','soundfile']}}
    dest = root / 'experiments/baselines/audioldm'; dest.mkdir(parents=True, exist_ok=True)
    (dest / (args.output_name+'_run.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__': main()
