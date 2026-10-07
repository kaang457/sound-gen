"""Numerical and trained-artifact checks; run with python -m unittest discover -s tests."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.control.spectral_cvae import descriptors, generate, ROOT, SR, N
from src.control.generate import generate as shared_generate
from src.control.tfoley import pulse_curve

class Controls(unittest.TestCase):
    def test_parseval_and_centroid_against_waveform(self):
        rng=np.random.default_rng(3);frames=rng.normal(size=(8,N))
        fft=abs(np.fft.rfft(frames));actual=descriptors(torch.tensor(fft,dtype=torch.float64)).numpy()
        np.testing.assert_allclose(actual[:,0],np.sqrt(np.mean(frames**2,axis=1)),rtol=1e-8)
        expected=(fft*np.fft.rfftfreq(N,1/SR)).sum(1)/fft.sum(1)
        np.testing.assert_allclose(actual[:,1],expected,rtol=1e-8)
    def test_unsupported_controls_and_invalid_curve(self):
        with self.assertRaises(ValueError):shared_generate('tfoley',ROOT,centroid_hz=2000)
        with self.assertRaises(ValueError):pulse_curve(onsets=[float('nan')])
    def test_trained_control_changes_and_determinism(self):
        y,sr,a=generate(ROOT,rms=.02,centroid_hz=1800,name='test_control')
        again,_,_=generate(ROOT,rms=.02,centroid_hz=1800,name='test_repeat')
        np.testing.assert_array_equal(y,again)
        _,_,b=generate(ROOT,rms=.04,centroid_hz=1800,name='test_louder')
        _,_,c=generate(ROOT,rms=.02,centroid_hz=2400,name='test_brighter')
        self.assertGreater(b['actual_mean_block_rms'],a['actual_mean_block_rms'])
        self.assertGreater(c['actual_mean_block_centroid_hz'],a['actual_mean_block_centroid_hz'])
        self.assertLess(abs(a['actual_mean_block_rms']/.02-1),.15)
        self.assertLess(abs(a['actual_mean_block_centroid_hz']-1800),150)
        for name in ['test_control','test_repeat','test_louder','test_brighter']:(ROOT/'results/audio/parametric'/f'{name}.wav').unlink()
if __name__=='__main__':unittest.main()
