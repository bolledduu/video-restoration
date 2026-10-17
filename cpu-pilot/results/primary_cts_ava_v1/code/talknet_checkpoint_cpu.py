"""Explicit checkpoint selection for the existing CPU TalkNet implementation."""
import hashlib,sys
from pathlib import Path


def load_detector(runtime_root,checkpoint):
    for folder in ('tools/quality_cpu','tools/iqa_models','tools/python','scripts'):sys.path.insert(0,str(Path(runtime_root)/folder))
    from evaluate_causal import TalkNetCPU
    detector=TalkNetCPU()
    import torch
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    detector.model.load_state_dict({k.removeprefix('model.'):v for k,v in state.items() if k.startswith('model.')},strict=True)
    for name,head in detector.heads.items():
        prefix='loss'+name+'.FC.';head.load_state_dict({k.removeprefix(prefix):v for k,v in state.items() if k.startswith(prefix)},strict=True)
    with Path(checkpoint).open('rb') as f:detector.checkpoint_sha256=hashlib.file_digest(f,'sha256').hexdigest()
    return detector
