"""Captured-support audit primitives. No labels, references or learned thresholds."""
import numpy as np
from scipy.ndimage import convolve

MASK_SEEDS=(1729,1730)
SHIFTS=(-3,3)
REGIONS={'lower_face':(.55,.90,.20,.80),'cheek':(.25,.50,.05,.65),'full_frame':(0.,1.,0.,1.)}

def region_mask(height,width,name):
    y0,y1,x0,x1=REGIONS[name];out=np.zeros((height,width),bool)
    out[int(y0*height):int(y1*height),int(x0*width):int(x1*width)]=True
    return out

def fixed_mask(height,width,seed,fraction=.10):
    if not 0<fraction<1:raise ValueError('Mask fraction must be between zero and one')
    result=np.zeros(height*width,bool)
    result[np.random.default_rng(seed).choice(height*width,int(round(height*width*fraction)),replace=False)]=True
    return result.reshape(height,width)

def masked_fill(captured,hidden):
    """Fill with retained spatial measurements only. Same hidden sites in every frame."""
    video=np.asarray(captured,dtype=np.float64)
    if video.ndim!=4 or video.shape[-1]!=3 or hidden.shape!=video.shape[1:3]:raise ValueError('Video/mask shape mismatch')
    keep=~hidden
    if not keep.any():raise ValueError('Cannot fill without retained observations')
    retained=np.where(keep[None,:,:,None],video,0.)
    # Fixed 5x5 spatial operator, no temporal borrowing or full-image statistics.
    kernel=np.ones((1,5,5,1),float)
    sums=convolve(retained,kernel,mode='constant',cval=0.)
    counts=convolve(keep[None,:,:,None].astype(float),kernel,mode='constant',cval=0.)
    fallback=retained.sum(axis=(1,2),keepdims=True)/keep.sum()
    fill=np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0)
    fill=np.where(counts>0,fill,fallback)
    return np.where(keep[None,:,:,None],video,fill)

def probes(baseline,restored,region,shifts=SHIFTS):
    """Use identical central time support; never circularly wrap temporal increments."""
    b=np.asarray(baseline,float);r=np.asarray(restored,float)
    if b.shape!=r.shape or b.ndim!=4:raise ValueError('Candidate grid mismatch')
    margin=max(abs(x) for x in shifts)
    if len(b)<=2*margin:raise ValueError('Clip too short for temporal controls')
    valid=np.arange(margin,len(b)-margin)
    delta=r-b;static=delta.mean(axis=0,keepdims=True);dynamic=delta-static
    output={'factual':r.copy()}
    for shift in shifts:
        v=r.copy()
        for t in valid:v[t,region]=b[t,region]+static[0,region]+dynamic[t+shift,region]
        output['shift_'+str(shift)]=v
    v=r.copy();v[:,region]=b[:,region]+static[:,region];output['static']=v
    # Same timing, changed magnitude: controls amplitude sensitivity.
    for scale in (.5,1.5):
        v=r.copy();v[:,region]=b[:,region]+static[:,region]+scale*dynamic[:,region]
        output['scale_'+str(scale)]=v
    return valid,output

def support_scores(captured,baseline,restored,hidden,region):
    valid,variants=probes(baseline,restored,region)
    sites=hidden & region
    if not sites.any():raise ValueError('No withheld measurements in region')
    y=np.asarray(captured,float)[valid][:,sites,:]/255.
    mse={name:np.mean((y-video[valid][:,sites,:]/255.)**2,axis=(1,2)) for name,video in variants.items()}
    contrast=np.mean([mse['shift_'+str(s)] for s in SHIFTS],axis=0)-mse['factual']
    return dict(indices=valid,mse=mse,temporal_support=contrast),variants

def quantize(video):return np.rint(np.clip(video,0,255)).astype(np.uint8)
def logits(p):
    p=np.clip(np.asarray(p,float),1e-5,1-1e-5);return np.log(p/(1-p))

def transition_targets(labels,captured,restored):
    y=np.asarray(labels);c=np.asarray(captured)>=.5;r=np.asarray(restored)>=.5
    if y.shape!=c.shape or y.shape!=r.shape or not np.isin(y,[0,1]).all():raise ValueError('Target grid mismatch')
    # CC=0, CI=1, IC=2, II=3; labels are only consumed by the evaluator/trainer.
    return np.where(c==y,np.where(r==y,0,1),np.where(r==y,2,3))
