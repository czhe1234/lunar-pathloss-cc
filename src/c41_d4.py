"""D4 action on C10's294 input descriptors. Geometry scalars stay invariant."""
import torch
def transform(x,k):
 if k==0:return x
 z=x.clone()
 for _ in range(k%4):
  old=z.clone()
  for u,v in [(3,4),(5,6),(34,35),(246,247),(266,267)]:
   z[:,u]=old[:,v];z[:,v]=-old[:,u]
  z[:,32]=255/256-old[:,33];z[:,33]=old[:,32]
  z[:,36]=old[:,37];z[:,37]=old[:,36];z[:,38]=-old[:,38]
  z[:,47:96]=torch.rot90(old[:,47:96].reshape(-1,7,7),1,(1,2)).reshape(-1,49)
 if k>=4:
  for u in [3,5,34,38,246,266]:z[:,u]=-z[:,u]
  z[:,33]=255/256-z[:,33]
  z[:,47:96]=torch.flip(z[:,47:96].reshape(-1,7,7),[2]).reshape(-1,49)
 return z
