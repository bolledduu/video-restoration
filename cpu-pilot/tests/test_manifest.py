import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import pilot

class PilotTests(unittest.TestCase):
    def test_three_unique_records(self):
        xs=pilot.MANIFEST["samples"];self.assertEqual(len(xs),3);self.assertEqual(len({x["video_id"] for x in xs}),3)
    def test_answers(self):
        for x in pilot.MANIFEST["samples"]:
            self.assertIn(x["published_answer"],"ABCD");self.assertEqual(len(x["choices"]),4)
    def test_not_a_restoration_claim(self):
        self.assertNotIn("audio_conditioned_restored",["degraded","sharpen_only"])
    def test_missing_files_remain_untested(self):
        with tempfile.TemporaryDirectory() as d:
            r=pilot.evaluate(d,Path(d)/"result.json");self.assertEqual(r["research_status"],"UNTESTED");self.assertFalse(r["annotation_valid"])
    def test_incomplete_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"pred.csv";p.write_text("sample_id,condition,prediction\npilot_01_speaker_recognition,degraded,D\n")
            with self.assertRaises(ValueError):pilot.score(p)
if __name__=="__main__":unittest.main()
