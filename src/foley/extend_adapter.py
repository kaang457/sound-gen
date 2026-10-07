"""Append an encoder without changing existing weights or claiming new control."""
from src.foley.controls import ControlSpec,ControlBank
from src.foley.adapter import EpsilonAdapter

def extend_checkpoint(state,new_spec):
    registry=state['registry']
    bank=ControlBank([ControlSpec(**x) for x in registry['controls']],registry['width'])
    adapter=EpsilonAdapter(bank);adapter.load_state_dict(state['state_dict'],strict=True)
    bank.add(new_spec)
    return {**state,'registry':bank.manifest(),'state_dict':adapter.state_dict(),
            'trained_controls':list(state.get('trained_controls',[])),
            'new_untrained_controls':list(state.get('new_untrained_controls',[]))+[new_spec.name]}

def freeze_for_extension(adapter,new_controls):
    """Train new encoders only; fixed trunk/old encoders preserve omitted-new outputs.

Only batches containing a new control have a gradient path in this mode.
This is a structural retention strategy, not proof of new-control quality.
"""
    if not new_controls or set(new_controls)-set(adapter.bank.encoders):raise ValueError('Unknown/empty new controls')
    adapter.requires_grad_(False)
    for name in new_controls:adapter.bank.encoders[name].requires_grad_(True)
    return [p for p in adapter.parameters() if p.requires_grad]
