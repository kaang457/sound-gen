"""Prototype residual epsilon adapter for AudioLDM. Requires Foley training.

Frozen backbone predicts epsilon; a learned residual receives noisy latents,
normalized diffusion time and identity-aware control fields. Zero output layer
preserves the pretrained prediction before training; it does not provide control.
"""
import torch
from torch import nn

class EpsilonAdapter(nn.Module):
    def __init__(self,bank,latent_channels=8,hidden=32):
        super().__init__();self.bank=bank
        self.semantic_projection=nn.Linear(512,bank.width)
        self.layers=nn.Sequential(nn.Conv2d(latent_channels+bank.width+1,hidden,3,padding=1),nn.SiLU(),nn.Conv2d(hidden,hidden,3,padding=1),nn.SiLU(),nn.Conv2d(hidden,latent_channels,1))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
    def forward(self,noisy,t,requests,duration_s,semantic_condition=None):
        if noisy.ndim!=4 or len(requests)!=len(noisy):raise ValueError('Latent/request shape mismatch')
        field,present=self.bank(requests,noisy.shape[2],duration_s)
        if semantic_condition is not None:
            if semantic_condition.shape!=(len(noisy),1,512):raise ValueError('Expected CLAP condition [B,1,512]')
            field=field+self.semantic_projection(semantic_condition[:,0]).unsqueeze(-1)
        field=field[:,:,:,None].expand(-1,-1,-1,noisy.shape[3])
        clock=t.to(noisy).reshape(-1,1,1,1)/999
        if clock.shape[0]!=len(noisy):raise ValueError('Timestep batch mismatch')
        clock=clock.expand(-1,1,noisy.shape[2],noisy.shape[3])
        residual=self.layers(torch.cat([noisy,field,clock],1))
        return residual*present[:,None,None,None].to(residual)

def training_loss(model,adapter,clean,condition,requests,duration_s,t=None,noise=None):
    """Actual pretrained denoiser is frozen; optimize only residual epsilon MSE."""
    if getattr(model,'parameterization',None)!='eps':raise ValueError('Only epsilon-prediction AudioLDM is supported')
    if t is None:t=torch.randint(0,model.num_timesteps,(len(clean),),device=clean.device)
    if noise is None:noise=torch.randn_like(clean)
    with torch.no_grad():
        noisy=model.q_sample(clean,t,noise=noise)
        baseline=model.apply_model(noisy,t,condition)
    residual=adapter(noisy,t,requests,duration_s,condition)
    loss=(baseline+residual-noise).float().square().mean()
    return loss,{'baseline_loss':float((baseline-noise).float().square().mean()),'residual_rms':float(residual.detach().square().mean().sqrt())}
