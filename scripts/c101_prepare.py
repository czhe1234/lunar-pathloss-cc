"""Prepare a direct frozenC82 residual experiment. H/TX geometry is separate from supervision."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c101_residual';D=R.parent/'data'
def mm(p):return np.load(p,mmap_mode='r')
def extra(mu):
 values=[(16*mu).clip(-4,4),np.arcsinh(64*mu)/4,np.arcsinh(4096*mu)/8,np.log10(abs(mu)+1e-8)/8]
 values += [np.tanh(mu/t) for t in [1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]]
 return np.stack(values,2).reshape(len(mu),48).astype('float32')
def geometry(h,tr,tc,px,old):
 rr,cc=np.divmod(px,256);v=h[rr,cc].astype('float64');tx=float(h[tr,tc]);z=float(np.floor(h[min(tr+1,255),tc])+3)
 dx=cc-tc;dy=rr-tr;dz=v+1-z;dist=np.sqrt(dx*dx+dy*dy+dz*dz).clip(.25)
 xp=h[rr,(cc+1).clip(0,255)].astype('float64')-v;xm=v-h[rr,(cc-1).clip(0,255)]
 yp=h[(rr+1).clip(0,255),cc].astype('float64')-v;ym=v-h[(rr-1).clip(0,255),cc]
 sx=np.stack([xp,xp,xm,xm],1);sy=np.stack([yp,ym,yp,ym],1)
 precise=(dz[:,None]-dx[:,None]*sx-dy[:,None]*sy)/(dist[:,None]*np.sqrt(1+sx*sx+sy*sy))
 p=old[:,:49].astype('float64').reshape(-1,7,7);cxp=p[:,3,4]-p[:,3,3];cxm=p[:,3,3]-p[:,3,2];cyp=p[:,4,3]-p[:,3,3];cym=p[:,3,3]-p[:,2,3]
 csx=np.stack([cxp,cxp,cxm,cxm],1);csy=np.stack([cyp,cym,cyp,cym],1);ray=old[:,98:101].astype('float64')
 coarse=(ray[:,2,None]-ray[:,0,None]*csx-ray[:,1,None]*csy)/np.sqrt(1+csx*csx+csy*csy)
 common=np.delete(old.astype('float32'),[102,103],axis=1)
 # Shared raw inputs retain released precision in both arms; only derived-incidence precision differs.
 rgrid=(rr[:,None]+np.repeat(np.arange(-3,4),7)).clip(0,255);cgrid=(cc[:,None]+np.tile(np.arange(-3,4),7)).clip(0,255)
 common[:,:49]=(h[rgrid,cgrid]-v[:,None]).clip(-10,10)
 common[:,96:98]=20*np.log10(dist[:,None]*np.array([415e6,5.775e9])[None]*4*np.pi/299792458.)
 common[:,98:101]=np.column_stack([dx/dist,dy/dist,dz/dist]);common[:,101]=np.log1p(dist)
 absolute=np.column_stack([v/100,np.full(len(v),tx/100),(v-tx)/30,rr/128,cc/128,np.full(len(v),tr/128),np.full(len(v),tc/128),np.log1p(dist)/6,
  np.remainder(v,1)-.5,np.full(len(v),np.remainder(tx,1)-.5),np.full(len(v),(tx-(z-3))/3)])
 common=np.c_[common,absolute].astype('float32')
 return common,extra(coarse),extra(precise),coarse,precise
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--join',action='store_true');ap.add_argument('--confirm',action='store_true');a=ap.parse_args()
 rows=list(csv.DictReader(open(D/'train_index.csv')));hh=mm(D/'train_hm.npy')
 exmap=mm(R/'features/c38_dense/fit_map_indices.npy');ex=mm(R/'features/c38_dense/fit_pixel.npy').reshape(len(exmap),-1);exidx={int(i):j for j,i in enumerate(exmap)}
 panel=np.load(R/'audit/c78_clear/train_panel.npz');dp={int(i):j for j,i in enumerate(panel['ids'])}
 src=R/'audit/c90_facet';sources={'fit':sorted(src.glob('fit[0-9]')),'monitor':sorted(src.glob('cal[0-9]'))+sorted(src.glob('select[0-9]'))}
 if a.confirm:sources={'confirm':sorted(src.glob('confirm[0-9]'))}
 if not a.join:
  for group,folders in sources.items():
   out=A/group;out.mkdir(parents=True,exist_ok=True)
   if (out/'geometry_done.json').exists():continue
   meta=[json.loads((f/'done.json').read_text()) for f in folders]
   allids=sum([m['ids'] for m in meta],[]);pp=1536 if group=='fit' else 2048;n=len(allids)*pp
   assert len(set(allids))==len(allids)
   arrays={k:np.lib.format.open_memmap(out/(k+'.npy'),mode='w+',dtype=dtype,shape=shape) for k,dtype,shape in [
    ('common','float32',(n,121)),('coarse','float32',(n,48)),('precise','float32',(n,48)),('target','float32',(n,4)),('base','float32',(n,2)),
    ('pixel','int32',(n,)),('mapid','int32',(n,)),('weight','float32',(n,))]}
   ts=time.monotonic();pos=0;signflip=0;near=0
   for folder,m in zip(folders,meta):
    xx=mm(folder/'x.npy');yy=mm(folder/'targets.npy')
    for j,i in enumerate(m['ids']):
     row=rows[i];h=hh[int(row['hm_row'])];tr=int(row['tx_row']);tc=int(row['tx_col'])
     if group=='fit':
      allowed=np.ones(65536,bool);allowed[ex[exidx[i]].astype('int64')]=False
      px=np.random.default_rng([90681,i]).choice(np.flatnonzero(allowed),pp,replace=False)
     else:px=panel['pixels'][dp[i]].astype('int64')
     common,coarse,precise,c,p=geometry(h,tr,tc,px,xx[j])
     sl=slice(pos*pp,(pos+1)*pp)
     arrays['common'][sl]=common;arrays['coarse'][sl]=coarse;arrays['precise'][sl]=precise
     arrays['target'][sl]=yy[j,:,:4];arrays['base'][sl]=yy[j,:,4:6] if group!='fit' else np.nan
     arrays['pixel'][sl]=px;arrays['mapid'][sl]=i;arrays['weight'][sl]=1.
     signflip+=int(((c>0)!=(p>0)).any(1).sum());near+=int((abs(p).min(1)<1e-3).sum())
     if pos==0:
      for b,suffix in enumerate(['415','58']):
       raw=mm(D/f'train_pl{suffix}.npy');assert np.array_equal(np.array(raw[i]).reshape(-1)[px].astype('float32'),yy[j,:,b])
     pos+=1
     if pos%250==0:print('geometry',group,pos,len(allids),'seconds',round(time.monotonic()-ts,1),flush=True)
   for v in arrays.values():v.flush()
   (out/'geometry_done.json').write_text(json.dumps(dict(ids=allids,points_per_map=pp,points=n,terrains=len({rows[i]['terrain_id'] for i in allids}),
    coarse_vs_precise_signflip_points=signflip,near_grazing_points=near,seconds=time.monotonic()-ts))+'\n')
  return
 # Join the independently generated input-only C82 fields; replace all C54 centre/edge predictors.
 bases={}
 for p in A.glob('base*of*'):
  if not (p/'done.json').exists():continue
  ar=mm(p/'prediction.npy');bases.update({int(i):(ar,j) for j,i in enumerate(json.loads((p/'indices.json').read_text()))})
 info=json.loads((R/'audit/c88_mesh/base_index.json').read_text());devbase=mm(R/'audit/c88_mesh/base.npy');devlookup={int(i):j for j,i in enumerate(info['ids'])}
 for group in sources:
  out=A/group;meta=json.loads((out/'geometry_done.json').read_text());ids=meta['ids'];pp=meta['points_per_map'];px=mm(out/'pixel.npy').reshape(len(ids),pp)
  x=np.load(out/'common.npy',mmap_mode='r+');base=np.load(out/'base.npy',mmap_mode='r+')
  for j,i in enumerate(ids):
   p=bases[i][0][bases[i][1]] if group=='fit' else devbase[devlookup[i]]
   rr,cc=np.divmod(px[j],256);b=p[:,rr,cc].T;sl=slice(j*pp,(j+1)*pp);x[sl,94:96]=b
   x[sl,102:110]=np.concatenate([(p[:,(rr+dr).clip(0,255),(cc+dc).clip(0,255)].T-b)/6 for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]],1)
   base[sl]=b
  x.flush();base.flush();(out/'done.json').write_text(json.dumps(meta)+'\n');print('joined',group,len(ids),flush=True)
 if not a.confirm:
  fit=A/'fit';meta=json.loads((fit/'done.json').read_text());pp=meta['points_per_map'];ids=meta['ids']
  from collections import Counter
  allfit=mm(R/'features/c38_dense/fit_map_indices.npy');count=Counter(rows[int(i)]['terrain_id'] for i in allfit);sel=Counter(rows[i]['terrain_id'] for i in ids)
  weight=np.load(fit/'weight.npy',mmap_mode='r+')
  for j,i in enumerate(ids):weight[j*pp:(j+1)*pp]=count[rows[i]['terrain_id']]/sel[rows[i]['terrain_id']]
  weight.flush()
  pick=(np.arange(len(ids))[:,None]*pp+np.arange(32)[None]).reshape(-1)
  x=np.c_[mm(fit/'common.npy')[pick],mm(fit/'precise.npy')[pick]].astype('float64')
  mean=x.mean(0);std=x.std(0).clip(.001);np.savez(A/'normalization.npz',mean=mean.astype('float32'),std=std.astype('float32'),edges=np.zeros((169,17),'float32'),target_mean=np.zeros(2,'float32'))
  print('normalization points',len(x),flush=True)
if __name__=='__main__':main()
