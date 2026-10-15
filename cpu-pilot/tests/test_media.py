import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import av
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cpu_media
import pilot

def video(path,count):
    with av.open(str(path),'w') as c:
        s=c.add_stream('libx264',rate=25);s.width=s.height=32;s.pix_fmt='yuv420p'
        for i in range(count):
            f=av.VideoFrame.from_ndarray(np.full((32,32,3),i*10,dtype=np.uint8),format='rgb24')
            for p in s.encode(f):c.mux(p)
        for p in s.encode():c.mux(p)

class MediaTests(unittest.TestCase):
    def test_truncated_candidate_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.mp4',Path(d)/'b.mp4';video(a,5);video(b,3)
            with self.assertRaisesRegex(ValueError,'count mismatch'):cpu_media.metrics(a,b)
            with self.assertRaisesRegex(ValueError,'frame count'):pilot.image_metrics(a,b)

    def test_identical_full_clip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.mp4';video(p,5);r=cpu_media.metrics(p,p)
            self.assertEqual(r['frames_evaluated'],5)
            self.assertEqual(r['gray_ssim_mean'],1.0)
            self.assertEqual(r['rgb_rmse'],0.0)

    def test_dataset_path_does_not_verify_annotations(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(pilot,'ROOT',Path(d)):
                r=pilot.evaluate(output_path=Path(d)/'result.json')
            self.assertFalse(r['annotation_valid'])

if __name__=='__main__':unittest.main()
