import unittest
import torch
from src.foley.controls import ControlSpec,ControlBank
from src.foley.adapter import EpsilonAdapter

class FoleyConditioning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.bank=ControlBank([ControlSpec('rms','scalar','linear',0,.5),ControlSpec('material','categorical','category',categories=('wood','metal'),families=('impact',))],8)
    def test_absence_is_not_zero_and_extension_preserves_existing(self):
        empty,_=self.bank([{'family':'impact','controls':{}}],16,5)
        zero,_=self.bank([{'family':'impact','controls':{'rms':0}}],16,5)
        self.assertEqual(torch.count_nonzero(empty),0);self.assertGreater(torch.count_nonzero(zero),0)
        request=[{'family':'impact','controls':{'rms':.03}}];before,_=self.bank(request,16,5)
        self.bank.add(ControlSpec('new_size','scalar','m',0,2))
        after,_=self.bank(request,16,5);torch.testing.assert_close(before,after,rtol=0,atol=0)
    def test_invalid_or_inapplicable_controls_rejected(self):
        for request in [{'family':'footstep','controls':{'material':'wood'}},{'family':'impact','controls':{'force':1}},{'family':'impact','controls':{'rms':None}},{'family':'impact','controls':{'rms':float('nan')}}]:
            with self.assertRaises(ValueError):self.bank([request],16,5)
    def test_curve_endpoint_and_range_validation(self):
        self.bank.add(ControlSpec('brightness','curve','Hz',0,8000))
        valid={'family':'impact','controls':{'brightness':{'times_s':[0,5],'values':[1000,3000]}}}
        field,_=self.bank([valid],16,5);self.assertFalse(torch.equal(field[:,:,0],field[:,:,-1]))
        for times,values in [([1,5],[1,2]),([0,5],[1,9000]),([0,0],[1,2])]:
            with self.assertRaises(ValueError):self.bank([{'family':'impact','controls':{'brightness':{'times_s':times,'values':values}}}],16,5)
    def test_zero_adapter_and_missing_control_after_updates(self):
        adapter=EpsilonAdapter(self.bank,hidden=8);x=torch.randn(1,8,16,4);t=torch.tensor([500])
        r=[{'family':'impact','controls':{'rms':.03}}]
        self.assertEqual(torch.count_nonzero(adapter(x,t,r,5)),0)
        optimizer=torch.optim.Adam(adapter.parameters(),lr=.01)
        loss=(adapter(x,t,r,5)-torch.ones_like(x)).square().mean();loss.backward();optimizer.step()
        self.assertGreater(torch.count_nonzero(adapter(x,t,r,5)),0)
        self.assertEqual(torch.count_nonzero(adapter(x,t,[{'family':'impact','controls':{}}],5)),0)
        before=adapter(x,t,r,5).detach().clone()
        self.bank.add(ControlSpec('size','scalar','m',0,2))
        torch.testing.assert_close(before,adapter(x,t,r,5),rtol=0,atol=0)
        from src.foley.extend_adapter import freeze_for_extension
        optimizer=torch.optim.Adam(freeze_for_extension(adapter,['size']),lr=.01)
        extended=[{'family':'impact','controls':{'rms':.03,'size':1.2}}]
        loss=(adapter(x,t,extended,5)-torch.ones_like(x)).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        torch.testing.assert_close(before,adapter(x,t,r,5),rtol=0,atol=0)
if __name__=='__main__':unittest.main()
