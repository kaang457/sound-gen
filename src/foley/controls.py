"""Typed, append-only control registry. Names are model inputs, not text prompts."""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
from torch import nn

@dataclass(frozen=True)
class ControlSpec:
    name: str
    kind: str
    unit: str
    minimum: float = 0
    maximum: float = 1
    categories: tuple = ()
    families: tuple = ()
    label_source: str = 'unavailable'
    def __post_init__(self):
        if not self.name or '.' in self.name: raise ValueError('Control name must be nonempty and contain no dot')
        if self.kind not in ('scalar','curve','categorical'): raise ValueError('Invalid control kind')
        if self.kind=='categorical':
            if not self.categories or len(set(self.categories))!=len(self.categories):raise ValueError('Unique categories required')
        elif not math.isfinite(self.minimum) or not math.isfinite(self.maximum) or self.minimum>=self.maximum:raise ValueError('Invalid control range')

class ControlBank(nn.Module):
    """Separate encoder per identity; adding absent controls preserves old outputs."""
    def __init__(self,specs,width=32):
        super().__init__();self.width=width;self.specs={};self.encoders=nn.ModuleDict()
        for spec in specs:self.add(spec)
    def add(self,spec):
        if spec.name in self.specs:raise ValueError('Registry entries cannot be overwritten')
        self.specs[spec.name]=spec
        module=(nn.Embedding(len(spec.categories),self.width) if spec.kind=='categorical' else nn.Sequential(nn.Linear(1,self.width),nn.SiLU(),nn.Linear(self.width,self.width)))
        # Match existing device/dtype when extending after model placement.
        previous=next(self.parameters(),None)
        if previous is not None:module=module.to(device=previous.device,dtype=previous.dtype)
        self.encoders[spec.name]=module
    def manifest(self):return {'format':'foley-controls-v1','width':self.width,'controls':[asdict(s) for s in self.specs.values()]}
    def forward(self,requests,frames,duration_s):
        if not isinstance(frames,int) or frames<1 or not math.isfinite(duration_s) or duration_s<=0:raise ValueError('Invalid duration/grid')
        ref=next(self.parameters());batch=[];present=[]
        for request in requests:
            family=request['family'];values=request.get('controls',{})
            if set(values)-set(self.specs):raise ValueError(f'Unknown controls: {set(values)-set(self.specs)}')
            output=torch.zeros(self.width,frames,device=ref.device,dtype=ref.dtype);count=0
            for name in self.specs:
                if name not in values:continue
                value=values[name]
                spec=self.specs[name]
                if spec.families and family not in spec.families:raise ValueError(f'{name} is not applicable to {family}')
                if value is None:raise ValueError('Omit absent controls; None is not zero')
                if spec.kind=='categorical':
                    if value not in spec.categories:raise ValueError(f'Invalid category for {name}')
                    idx=torch.full((frames,),spec.categories.index(value),device=ref.device,dtype=torch.long)
                    encoded=self.encoders[name](idx)
                else:
                    if spec.kind=='curve':
                        if not isinstance(value,dict) or set(value)!= {'times_s','values'}:raise ValueError('Curve requires times_s and values')
                        times=np.asarray(value['times_s'],float);raw=np.asarray(value['values'],float)
                        if times.ndim!=1 or raw.shape!=times.shape or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0) or times[0]!=0 or not np.isclose(times[-1],duration_s):raise ValueError('Curve must cover [0,duration] with strictly increasing times')
                        if not np.isfinite(raw).all() or np.any((raw<spec.minimum)|(raw>spec.maximum)):raise ValueError(f'Out of range: {name}')
                        raw=np.interp(np.linspace(0,duration_s,frames),times,raw)
                    else:
                        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not spec.minimum<=value<=spec.maximum:raise ValueError(f'Invalid numeric value: {name}')
                        raw=np.full(frames,value)
                    normalized=2*(raw-spec.minimum)/(spec.maximum-spec.minimum)-1
                    tensor=torch.tensor(normalized[:,None],device=ref.device,dtype=ref.dtype)
                    encoded=self.encoders[name](tensor)
                output=output+encoded.T;count+=1
            batch.append(output);present.append(bool(count))
        if not batch:raise ValueError('Empty batch')
        return torch.stack(batch),torch.tensor(present,device=ref.device)
