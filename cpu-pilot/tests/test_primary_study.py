import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_primary_cts_study import normalize_training

class PrimaryStudyTests(unittest.TestCase):
    def test_training_uses_full_source_person_time_identity_and_checkpoint(self):
        roster={'cases':[dict(id='canonical',source='source',entity='person',start=1.,split='train')]}
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'input.json';p.write_text(json.dumps(dict(status='COMPLETE',detector_checkpoint_sha256='expected',cases=[dict(id='old',source='source',entity='person',start=1.)])))
            data=normalize_training(roster,[p],'expected')
            self.assertEqual(data['cases'][0]['id'],'canonical')
            self.assertEqual(data['cases'][0]['original_case_id'],'old')
            with self.assertRaisesRegex(ValueError,'checkpoint'):normalize_training(roster,[p],'different')
            with self.assertRaisesRegex(ValueError,'duplicated'):normalize_training(roster,[p,p],'expected')
    def test_missing_planned_training_case_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'input.json';p.write_text(json.dumps(dict(status='COMPLETE',detector_checkpoint_sha256='expected',cases=[])))
            with self.assertRaisesRegex(ValueError,'incomplete'):normalize_training({'cases':[dict(id='a',source='s',entity='p',start=1.,split='train')]},[p],'expected')
if __name__=='__main__':unittest.main()
