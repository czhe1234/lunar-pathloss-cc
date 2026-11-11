"""Paired check of receiver-intersection origin offsets. Diagnostic only; no new training targets."""
import csv,ctypes,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,DATA,atomic_json
from src.c69_registered import RegisteredTransport
from src.c67_physics import quadrature
from src.c67_dense import Panel
from src.c72_score_dev import load_net
import numpy as np
import torch
@torch.inference_mode()
def main():
    g=gpu_guard();source=R/'audit/c72_continuation';rng=np.random.default_rng(77677)
    panels=json.load(open(R/'audit/c66_transport/panels.json'));protocol=json.load(open(R/'runs/c72_observed/protocol.json'))
    rows=list(csv.DictReader((DATA/'train_index.csv').open()));hh=np.load(DATA/'train_hm.npy',mmap_mode='r')
    fit=rng.choice(panels['fit_first'],64,replace=False).tolist()
    same={}
    for i in protocol['confirmation']:same.setdefault(rows[i]['terrain_id'],i)
    dev=list(same.values());assert len(dev)==64
    eng=RegisteredTransport(physical=1,anti=0,smooth=1,direct=2,spec=4,diffuse=128);lib=ctypes.CDLL(str(R/'src/libc77_offset.so'))
    P=ctypes.c_void_p;S=ctypes.c_float;I=ctypes.c_int;fn=lib.offset77;fn.argtypes=[P,P,I,S,S,S,S,S,P,P];fn.restype=I
    q=quadrature(4);q2=quadrature(6)
    def pairs(h,tr,tc,quad):
        height=torch.from_numpy(np.array(h,dtype=np.float32,copy=True)).cuda()
        geom,_=eng.scene(height);out=torch.zeros((len(eng.faces),2),device='cuda')
        code=fn(geom.data_ptr(),quad.data_ptr(),len(quad),tc*255/256,(tr+1)*255/256,float(np.floor(h[min(tr+1,255),tc])+3),256/255,float(h.max()),out.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError(code)
        return out
    flat=np.zeros((256,256),np.float32);reference=pairs(flat,128,128,q)
    relative=((reference[:,0]-reference[:,1]).abs()/reference[:,0].clamp_min(1e-12)).amax().item()
    assert relative<1e-5,relative
    model=load_net(R/'runs/c72_observed')
    accum=[];sensitivity=[]
    tic=time.monotonic()
    for part,ids in [('fit',fit),('dev',dev)]:
        panel=Panel(ids,part,'features/c72_native');records=[]
        for j,i in enumerate(ids):
            row=rows[i];r=int(row['tx_row']);c=int(row['tx_col']);h=hh[int(row['hm_row'])]
            with torch.autocast('cuda',torch.bfloat16):corr=model(panel.x[j:j+1].float())[0].float()
            err=panel.y[j]-corr;w=panel.w[j]
            direct,spec,diff,*_=eng.raw(h,r,c);bg=spec[:,:2]+diff[:,:2]
            def db(quad):
                pred=pairs(h,r,c,quad);zero=pred[:,0:1];shift=pred[:,1:2]
                # Use paired correction on the same native quadrature, with the same first-order background.
                v=-10*(torch.log10((shift+bg).clamp_min(1e-25))-torch.log10((zero+bg).clamp_min(1e-25)))
                return eng.interpolate(v).T.reshape(2,256,256).clamp(-30,30)
            d=db(q)
            if j<4:
                dd=db(q2);sensitivity.append(dict(part=part,row=i,delta_RMS=d.square().mean((1,2)).sqrt().tolist(),quadrature_difference=(d-dd).square().mean((1,2)).sqrt().tolist()))
            vis=eng.interpolate(direct[:,2:3]).reshape(256,256);zone=torch.where(vis>.99,0,torch.where(vis>=.01,1,2))
            loc=[]
            for k in range(3):
                m=w&(zone==k)
                loc.append(torch.stack([(e*m).sum((1,2),dtype=torch.float64) for e in [err.square(),err*d,d.square(),torch.ones_like(err)]],-1).cpu().numpy())
            records.append(np.stack(loc,axis=1))
            if j%16==0:print(part,j,time.monotonic()-tic,flush=True)
        accum.append(np.stack(records));del panel
    fitstats,devstats=accum;a=fitstats.sum(0);beta=np.clip(a[...,1]/np.maximum(a[...,2],1e-12),0,1)
    b=devstats.sum(0);old=b[...,0];new=old-2*beta*b[...,1]+beta**2*b[...,2]
    def rms(ss,ct):return np.sqrt(ss/ct)
    oldrm=rms(old.sum(1),b[...,3].sum(1));newrm=rms(new.sum(1),b[...,3].sum(1))
    rec=dict(scope='64 FIT terrains for beta, 64 held-out internal confirmation terrains, one full map each. Official VAL unused. Both powers are quadrature approximations on an inferred receiver grid.',
        reference='C72 observed',gpu=g,eps=8.94069671631e-5,mode='centered 256m candidate coordinates, receiver origin offset after each measurement hit, LOS rays only, shared reflection background',
        flat_relative_max_difference=relative,beta=beta.tolist(),raw_rmse=oldrm.tolist(),corrected_rmse=newrm.tolist(),gain=(oldrm-newrm).tolist(),zone_rmse_original=rms(old,b[...,3]).tolist(),zone_rmse_corrected=rms(new,b[...,3]).tolist(),
        numerical_sensitivity=sensitivity,seconds=time.monotonic()-tic)
    atomic_json(source/'c77_offset_result.json',rec);np.savez(source/'c77_offset_pair.npz',fit=fitstats,dev=devstats,fit_rows=fit,dev_rows=dev);print(json.dumps(rec),flush=True)
if __name__=='__main__':main()
