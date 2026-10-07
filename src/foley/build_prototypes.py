"""Offline audio semantic bank from train recordings only; no text encoder."""
import argparse
import json
from pathlib import Path
import sys
import torch
from torch.nn.functional import normalize
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.foley.prepare_cache import validate_manifest

def build(root,index_path):
    rows=validate_manifest(json.loads(Path(index_path).read_text()));events={}
    for row in rows:
        if row['split']!='train':continue
        cached=torch.load(row['cache_path'],map_location='cpu',weights_only=True)
        group=events.setdefault(row['event_id'],{'family':row['family'],'recordings':{},'sources':[]})
        group['recordings'].setdefault(row['recording_id'],[]).append(cached['condition'])
        group['sources'].append({k:row[k] for k in ['recording_id','license','source_url','sha256']})
    output=Path(root)/'experiments/architecture';output.mkdir(parents=True,exist_ok=True);written=[]
    for event,group in events.items():
        if len(group['recordings'])<2:raise ValueError('A production prototype requires >=2 distinct training recordings; single-exemplar smoke stays separate')
        per_record=[torch.stack(items).mean(0) for items in group['recordings'].values()]
        condition=normalize(torch.stack(per_record).mean(0),dim=-1)
        report={'event_id':event,'family':group['family'],'condition':condition.tolist(),'support_count':len(group['recordings']),'sources':group['sources'],'conditioning':'normalized mean CLAP audio embeddings; equal weight per recording; train-only','model_revision':'4054cb418ba3d947d94f6ad302f1d6a320e2313c'}
        path=output/f'{event}_prototype.json';path.write_text(json.dumps(report,indent=2));written.append(str(path))
    if not written:raise ValueError('No train prototypes')
    return written
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);a=p.parse_args();print(build(ROOT,a.index))
