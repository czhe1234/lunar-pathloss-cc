"""Deterministic C10 descriptors, exactly the formula used in the point data."""
import numpy as np
def phase_features(v):
 t=(np.arange(128,dtype=np.float32)+.5)/128;tt=np.r_[np.float32(0),t,np.float32(1)];dt=np.diff(tt)
 frac=np.remainder(v[:,31],1);e=2*np.sinh(v[:,104:232]*3)+frac[:,None]*(1-t)
 bins=e.reshape(-1,8,16).max(2);ep=np.concatenate((-(3-frac[:,None]),e,np.full((len(v),1),-1,np.float32)),1)
 a,z=ep[:,:-1],ep[:,1:];length=((np.maximum(a,0)+np.maximum(z,0))/np.maximum(abs(a)+abs(z),1e-9)*dt).sum(1)
 dist=np.sqrt((v[:,5]*128)**2+(v[:,6]*128)**2+(v[:,7]*50+frac)**2)
 extras=np.column_stack((np.arcsinh(bins/2)/3,np.log1p(length*dist)/6,(e>0).mean(1),t[e.argmax(1)],np.arcsinh((e/t).max(1)/2)/3,((a>0)!=(z>0)).sum(1),3-frac))
 return extras.astype(np.float32)

