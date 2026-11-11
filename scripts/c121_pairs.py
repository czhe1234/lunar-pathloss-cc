"""Same released terrain and TX, near RX opposite/same residual pairs. CPU only."""
import sys,json,csv,time,argparse
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.c101_prepare import geometry
A=R/'audit/c121_pairs';D=R.parent/'data'
def mm(p): return np.load(p,mmap_mode='r')
def index(root,filename='features.npy'):
 d={}
 for p in sorted((R/root).glob('part*')):
  if not (p/'complete.json').exists():continue
  arr=mm(p/filename)
  d.update({int(k):(arr,j) for j,k in enumerate(json.load(open(p/'indices.json')))})
 return d
class Reader:
 def __init__(self):
  self.rows=list(csv.DictReader(open(D/'train_index.csv')));self.H=mm(D/'train_hm.npy')
  self.compact=index('features/c73_compact');self.compact.update(index('features/c73_dev_extra_compact'))
  self.native=index('features/c72_native');self.native.update(index('features/c72_dev_extra_native'))
  fitids=json.load(open(R/'audit/c101_residual/base0of1/indices.json'))
  fb=mm(R/'audit/c101_residual/base0of1/prediction.npy')
  devids=json.load(open(R/'audit/c88_mesh/base_index.json'))['ids'];db=mm(R/'audit/c88_mesh/base.npy')
  self.base={int(i):(fb,j) for j,i in enumerate(fitids)};self.base.update({int(i):(db,j) for j,i in enumerate(devids)})
  self.Y=[mm(D/f'train_pl{k}.npy') for k in ['415','58']]
  self.M=[mm(D/f'train_mask{k}.npy') for k in ['415','58']]
  rec=np.load(R/'src/c70_centroid_lookup0.npz');self.donor=rec['ids'];self.dw=rec['weights']
 def scene(self,i):
  row=self.rows[i];return self.H[int(row['hm_row'])],int(row['tx_row']),int(row['tx_col'])
 def pred(self,i):
  arr,j=self.base[i];return arr[j]
 def old(self,i,px):
  h,tr,tc=self.scene(i);rr,cc=np.divmod(px,256);n=len(px);v=h[rr,cc].astype('float64')
  raw=np.zeros((n,112),np.float32)
  raw[:,:49]=(h[(rr[:,None]+np.repeat(np.arange(-3,4),7)).clip(0,255),
   (cc[:,None]+np.tile(np.arange(-3,4),7)).clip(0,255)]-v[:,None]).clip(-10,10)
  ar,j=self.compact[i];raw[:,49:94]=ar[j,:,rr,cc]
  p=self.pred(i);center=p[:,rr,cc].T;raw[:,94:96]=center
  raw[:,104:112]=np.concatenate([(p[:,(rr+dr).clip(0,255),(cc+dc).clip(0,255)].T-center)/6 for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]],1)
  common,_,precise,_,mu=geometry(h,tr,tc,px,raw)
  return np.c_[common,precise],mu
 def donors(self,i,px):
  ids=self.donor[px];wt=self.dw[px]
  order=np.argsort(-wt,axis=1,kind='stable');ids=np.take_along_axis(ids,order,1);wt=np.take_along_axis(wt,order,1)
  face=ids//65025;cell=ids%65025;rr,cc=np.divmod(cell,255)
  ar,j=self.native[i]
  out=np.stack([ar[j,face[:,k,None]*47+np.arange(47)[None],rr[:,k,None],cc[:,k,None]] for k in range(3)],1).astype('float32')
  # Same frozen receiver lookup as existing C72. No new mesh or spatial offsets.
  pr,pc=np.divmod(px,256);cx=cc+np.where(face==0,2/3,1/3);cy=rr+np.where(face==0,1/3,2/3)
  meta=np.concatenate([wt,cx-pc[:,None],cy-pr[:,None]],1).astype('float32')
  return out,meta

def describe(x):
 x=np.asarray(x);return {'n':int(x.size),'mean':float(x.mean()) if x.size else None,'median':float(np.median(x)) if x.size else None,
 'q10':float(np.quantile(x,.1)) if x.size else None,'q90':float(np.quantile(x,.9)) if x.size else None}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['fit','monitor'],default='monitor');ap.add_argument('--maxmaps',type=int,default=0);a=ap.parse_args()
 A.mkdir(exist_ok=True);rd=Reader();P=R/'audit/c101_residual'/a.part;info=json.load(open(P/'done.json'));pp=info['points_per_map']
 cp=mm(P/'pixel.npy');cy=mm(P/'target.npy');cb=mm(P/'base.npy');old0=mm(P/'common.npy');old1=mm(P/'precise.npy')
 norm=np.load(R/'audit/c101_residual/normalization.npz');std=norm['std']
 # Isolate geometry-only existing channels; C82 values are excluded from similarity.
 geomcols=np.r_[np.arange(94),np.arange(96,102),np.arange(110,169)]
 rows=[];features=[];donors=[];metas=[];counts=np.zeros((2,6),np.int64);parity=[];start=time.monotonic()
 dr,dc=np.meshgrid(np.arange(-4,5),np.arange(-4,5),indexing='ij');dr=dr.ravel();dc=dc.ravel();keep=(dr!=0)|(dc!=0);dr=dr[keep];dc=dc[keep]
 for mj,i in enumerate(info['ids'][:a.maxmaps or None]):
  sl=slice(mj*pp,(mj+1)*pp);pixels=np.array(cp[sl]);target=np.array(cy[sl]);base=np.array(cb[sl]);res=target[:,:2]-base
  testpx=pixels[:8];xx,_=rd.old(i,testpx);parity.append(float(np.max(np.abs(xx-np.c_[old0[sl][:8],old1[sl][:8]]))))
  if max(parity)>1e-5:raise AssertionError(('old reconstruction',i,max(parity)))
  h,tr,tc=rd.scene(i);allpred=rd.pred(i)
  for band in range(2):
   anchor=np.flatnonzero((target[:,2+band]>0)&(res[:,band]>=5))
   counts[band,0]+=len(anchor)
   if not len(anchor):continue
   pa=pixels[anchor];ar,ac=np.divmod(pa,256);rr=ar[:,None]+dr;cc=ac[:,None]+dc;inside=(rr>=0)&(rr<256)&(cc>=0)&(cc<256)
   pb=(rr.clip(0,255)*256+cc.clip(0,255));flat=pb.ravel()
   # Build once for all actual endpoints; gather repeated candidates.
   unique,inv=np.unique(np.r_[pa,flat],return_inverse=True);ux,um=rd.old(i,unique)
   na=len(pa);xa=ux[inv[:na]];xb=ux[inv[na:]].reshape(na,-1,169);ma=um[inv[:na]];mb=um[inv[na:]].reshape(na,-1,4)
   ya=np.asarray(rd.Y[band][i]).reshape(-1)[unique].astype('float32');pred=allpred[band].reshape(-1)[unique]
   valid=np.unpackbits(np.asarray(rd.M[band][i])).reshape(-1)[unique]>0
   re=(ya-pred)[inv[na:]].reshape(na,-1);va=valid[inv[na:]].reshape(na,-1)&inside
   patch=np.sqrt(np.mean((xb[:,:,:49]-xa[:,None,:49])**2,2))
   height=np.abs(xb[:,:,110]-xa[:,None,110])*100
   angle=np.arccos(np.clip(np.sum(xb[:,:,98:101]*xa[:,None,98:101],2),-1,1))
   rel=np.abs(np.expm1(xb[:,:,101])-np.expm1(xa[:,None,101]))/np.expm1(xa[:,None,101]).clip(.25)
   mud=np.sqrt(np.mean((mb-ma[:,None])**2,2))
   distance=np.stack([height/.25,patch/.1,angle/.025,rel/.03,mud/.01],2)
   tight=np.all(distance<=1,2);very=np.all(distance<=.5,2);cost=np.sqrt(np.mean(distance**2,2))
   opposite=va&(re<=-5);same=va&(re>=5)
   counts[band,1]+=opposite.any(1).sum();counts[band,2]+=(opposite&tight).any(1).sum();counts[band,3]+=(opposite&very).any(1).sum()
   for k in np.flatnonzero((opposite&tight).any(1)):
    kk=int(np.argmin(np.where(opposite[k]&tight[k],cost[k],np.inf)))
    rad=np.maximum(abs(dr),abs(dc));eligible=same[k]&tight[k]&(cost[k]>=.8*cost[k,kk])&(cost[k]<=1.25*cost[k,kk])&(abs(rad-rad[kk])<=1)
    ss=int(np.argmin(np.where(eligible,np.abs(cost[k]-cost[k,kk]),np.inf))) if eligible.any() else -1
    counts[band,4]+=ss>=0
    indices=[int(pa[k]),int(pb[k,kk]),int(pb[k,ss]) if ss>=0 else int(pa[k])]
    ii=np.searchsorted(unique,indices);fx=ux[ii];fy=np.array([rd.Y[b][i].reshape(-1)[indices] for b in range(2)]).T;fp=allpred.reshape(2,-1)[:,indices].T
    gdiff=np.sqrt(np.mean(((fx[1,geomcols]-fx[0,geomcols])/std[geomcols])**2))
    rec=dict(mapid=int(i),terrain_id=rd.rows[i]['terrain_id'],band=band,tx_row=tr,tx_col=tc,
      anchor_row=int(ar[k]),anchor_col=int(ac[k]),opposite_row=int(rr[k,kk]),opposite_col=int(cc[k,kk]),
      same_pixel=int(indices[2]) if ss>=0 else -1,anchor_pixel=int(indices[0]),opposite_pixel=int(indices[1]),
      anchor_cache_row=int(mj*pp+anchor[k]),cost=float(cost[k,kk]),same_cost=float(cost[k,ss]) if ss>=0 else None,
      height_delta=float(height[k,kk]),patch7_delta_rms=float(patch[k,kk]),ray_angle=float(angle[k,kk]),
      relative_distance=float(rel[k,kk]),incidence_delta_rms=float(mud[k,kk]),old_geometry_rms_z=float(gdiff),
      anchor_residual=float(fy[0,band]-fp[0,band]),opposite_residual=float(fy[1,band]-fp[1,band]),
      same_residual=float(fy[2,band]-fp[2,band]) if ss>=0 else None,
      target=fy.tolist(),base=fp.tolist(),very_tight=bool(very[k,kk]))
    rows.append(rec);features.append(fx)
    if i in rd.native:
     nv,meta=rd.donors(i,np.array(indices));donors.append(nv);metas.append(meta)
    else:donors.append(np.full((3,3,47),np.nan,np.float32));metas.append(np.full((3,9),np.nan,np.float32))
  if (mj+1)%50==0:print(a.part,mj+1,'pairs',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
 dst=A/a.part;dst.mkdir(exist_ok=True);(dst/'pairs.json').write_text(json.dumps(rows,ensure_ascii=False)+'\n')
 np.savez(dst/'pair_features.npz',old=np.array(features),donors=np.array(donors),metadata=np.array(metas))
 summary=dict(part=a.part,maps=mj+1,counts_columns=['positive_tail_anchors','has_near_opposite','has_tight_opposite','has_very_tight_opposite','tight_with_same_sign_control','unused'],
 counts=counts.tolist(),feature_reconstruction_max_abs=max(parity),pairs=len(rows),seconds=time.monotonic()-start)
 for b in range(2):
  rr=[r for r in rows if r['band']==b];summary[f'band{b}']={k:describe([r[k] for r in rr]) for k in ['cost','patch7_delta_rms','height_delta','ray_angle','incidence_delta_rms','old_geometry_rms_z','anchor_residual','opposite_residual']}
 (dst/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
