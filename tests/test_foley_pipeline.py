import json
from pathlib import Path
import unittest
from src.foley.prepare_cache import validate_manifest
from src.foley.infer import render,TrainingRequiredError,ROOT

class FoleyPipeline(unittest.TestCase):
    def test_recording_leakage_rejected(self):
        row={'recording_id':'same','path':'a.wav','event_id':'footstep','family':'footstep','license':'CC0-1.0','source_url':'https://example.org/a','split':'train'}
        with self.assertRaises(ValueError):validate_manifest({'records':[row,{**row,'split':'test'}]})
    def test_physical_label_requires_provenance(self):
        row={'recording_id':'a','path':'a.wav','event_id':'footstep','family':'footstep','license':'CC0-1.0','source_url':'https://example.org/a','split':'train','controls':{'force_n':200}}
        with self.assertRaises(ValueError):validate_manifest({'records':[row]})
    def test_numeric_request_not_silently_ignored(self):
        request=json.loads((ROOT/'configs/foley_target_request.json').read_text())
        with self.assertRaises(TrainingRequiredError):render(ROOT,request)
    def test_user_prompt_not_accepted(self):
        request=json.loads((ROOT/'configs/foley_baseline_request.json').read_text());request['prompt']='footstep'
        with self.assertRaises(ValueError):render(ROOT,request)
if __name__=='__main__':unittest.main()
