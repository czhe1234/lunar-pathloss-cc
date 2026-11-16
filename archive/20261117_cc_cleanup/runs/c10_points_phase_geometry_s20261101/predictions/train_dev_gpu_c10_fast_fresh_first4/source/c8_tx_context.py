"""Input-only transmitter context and raster-height phase descriptors."""
import numpy as np
from scipy.ndimage import uniform_filter
OFF=(-4,-2,-1,0,1,2,4)
NAMES=['TX_height','TX_row','TX_col','TX_slope_x','TX_slope_y','TX_slope_xx','TX_slope_yy','TX_slope_xy','terrain_mean','terrain_std','terrain_max','terrain_q10','terrain_q25','terrain_q50','terrain_q75','terrain_q90']+[f'TX_patch_{dy}_{dx}' for dy in OFF for dx in OFF]+[f'TX_height_{kind}_{q}' for q in [.25,.5,1,2] for kind in ['sin','cos']]
def terrain_fields(h):
 sy,sx=np.gradient(h);syy,syx=np.gradient(sy);sxy,sxx=np.gradient(sx)
 stats=np.r_[h.mean(),h.std(),h.max(),np.quantile(h,[.1,.25,.5,.75,.9])]
 return h,sx,sy,sxx,syy,sxy,stats
def tx_features(fields,tr,tc):
 h,sx,sy,sxx,syy,sxy,stats=fields
 vals=[h[tr,tc],tr/256,tc/256,sx[tr,tc],sy[tr,tc],sxx[tr,tc],syy[tr,tc],sxy[tr,tc],*stats]
 for dr in OFF:
  for dc in OFF:vals.append(h[np.clip(tr+dr,0,255),np.clip(tc+dc,0,255)]-h[tr,tc])
 for q in [.25,.5,1,2]:vals.extend([np.sin(2*np.pi*h[tr,tc]/q),np.cos(2*np.pi*h[tr,tc]/q)])
 return np.array(vals,np.float32)

