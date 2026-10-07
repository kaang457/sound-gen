"""Shared entry point; unsupported controls raise instead of being ignored."""
from pathlib import Path

def generate(model,root,**controls):
    if model=='tfoley':
        allowed={'rms','name','seed','steps','class_name','device','prepare'}
        if set(controls)-allowed:raise ValueError(f'Unsupported TFOLEY controls: {set(controls)-allowed}')
        from src.control.tfoley import generate as run
        return run(root,**controls)
    if model=='spectral_cvae':
        allowed={'rms','centroid_hz','duration_s','seed','name'}
        if set(controls)-allowed:raise ValueError(f'Unsupported CVAE controls: {set(controls)-allowed}')
        from src.control.spectral_cvae import generate as run
        return run(root,**controls)
    raise ValueError('Supported models: tfoley, spectral_cvae')
