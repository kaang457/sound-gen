"""Numeric controls for pinned TFOLEY; no user text or output gain matching."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np

CAPABILITIES = {'rms_curve': True, 'centroid_hz': False, 'pitch_hz': False,
                'duration_s': [4], 'classes': ['DogBark','Footstep','GunShot','Keyboard','MovingMotorVehicle','Rain','Sneeze_Cough']}

def pulse_curve(level=0.03, onsets=(0.6, 1.8, 3.0), decay_s=0.12):
    if not np.isfinite(level) or not 0 < level <= 0.2: raise ValueError('level must be in (0, .2]')
    if not np.isfinite(decay_s) or decay_s <= 0: raise ValueError('positive decay required')
    onsets=np.asarray(onsets,float)
    if onsets.ndim!=1 or not len(onsets) or not np.isfinite(onsets).all() or np.any((onsets<0)|(onsets>=4)): raise ValueError('onsets must lie in [0,4)')
    t=np.arange(690)*128/22050
    # Smooth pulse avoids a discontinuous target; level is peak RMS, not global RMS.
    return np.sum([level*np.exp(-0.5*((t-o)/decay_s)**2) for o in onsets],axis=0)

def generate(root, rms, name, seed=42, steps=50, class_name='Footstep', device='auto', prepare=False):
    root=Path(root).resolve()
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name): raise ValueError('Invalid output stem')
    if class_name not in CAPABILITIES['classes']: raise ValueError('Unsupported class')
    rms=np.asarray(rms,dtype=np.float32)
    if rms.shape!=(690,) or not np.isfinite(rms).all() or np.any(rms<0): raise ValueError('Expected 690 finite nonnegative RMS values')
    folder=root/'experiments/parametric/conditions';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(name+'.json');path.write_text(json.dumps({'rms':rms.tolist()}),encoding='utf-8')
    command=[sys.executable,str(root/'src/baselines/tfoley_inference.py'),'--rms-json',str(path),'--output-name',name,'--seed',str(seed),'--steps',str(steps),'--class-name',class_name,'--device',device]
    if prepare: command.append('--prepare')
    subprocess.run(command,check=True)
    return root/'results/audio/tfoley'/(name+'.wav')
