"""Do the terrain-selected facet network's tiny gains localize the actual clear weak points?"""
import json,sys
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c90_facet import A,Data,Facets,predict,gpu_guard
def main():
    gpu=gpu_guard();D=Data('confirm','cpu');arm=json.load(open(A/'choice.json'))['choice_train_monitor'];assert arm=='fixed'
    ck=torch.load(R/'runs'/('c90_'+arm)/'best.pt',weights_only=True,map_location='cpu');m=Facets(arm).cuda();m.load_state_dict(ck['state'])
    a=predict(m,D).reshape(-1,D.m,2);x=D.x.numpy().reshape(-1,D.m,112);t=D.t.numpy().reshape(-1,D.m,6)
    ref=np.load(R/'audit/c78_clear/train_reference.npy',mmap_mode='r');order={i:j for j,i in enumerate(json.load(open(R/'audit/c88_mesh/base_index.json'))['ids'])}
    geo={}
    for f in sorted((R/'audit/c78_clear').glob('train_geom_*of3')):
        ar=np.load(f/'geometry.npy',mmap_mode='r');geo.update({k:(ar,j) for j,k in enumerate(np.load(f/'positions.npy'))})
    s=np.zeros((2,2,6));ci=np.zeros((2,2));B=np.array(ck['beta'])
    for i,row in enumerate(D.ids):
        pos=order[row];w=t[i,:,2:4]>.5;y=t[i,:,:2];p=t[i,:,4:];v=ref[pos,:,6]
        vgate=x[i,:,65].astype('float32');z=(vgate<.01).astype(int)*2+((vgate>=.01)&(vgate<=.99))
        bins=2*z[:,None]+((p-x[i,:,96:98].astype('float32'))>3)
        pred=p+np.stack([B[b,bins[:,b]]*(a[i,:,b]-p[:,b]).clip(-30,30) for b in (0,1)],1)
        gh,j=geo[pos];radius=gh[j,:,17].astype('float64').clip(.001)
        fs=20*np.log10(radius[:,None]*np.array([415e6,5.775e9])*4*np.pi/299792458)
        for n,q in enumerate((p,pred)):
            for b in (0,1):
                k=w[:,b]&(v>.99);weak=(y[:,b]>fs[:,b]+3)&k;call=(q[:,b]>fs[:,b]+3)&k
                e=q[:,b]-y[:,b];s[n,b]+=[weak.sum(),call.sum(),(weak&call).sum(),(e[weak]**2).sum(),(e[k]**2).sum(),k.sum()]
            k=w.all(1)&(v<.01);di=(q[:,1]-q[:,0])-(y[:,1]-y[:,0]);ci[n]+=[k.sum(),(di[k]**2).sum(dtype='float64')]
    out={}
    for n,name in enumerate(('C82','C90_fixed_blend')):
        z=s[n];out[name]=dict(recall=(z[:,2]/z[:,0]).tolist(),precision=(z[:,2]/z[:,1]).tolist(),true_weak_RMSE=(z[:,3]/z[:,0])**.5,
             clear_RMSE=(z[:,4]/z[:,5])**.5,clear_valid=z[:,5].astype(int).tolist(),true_weak=z[:,0].astype(int).tolist(),shadow_joint=int(ci[n,0]),contrast=float(np.sqrt(ci[n,1]/ci[n,0])))
        out[name]['clear_RMSE']=out[name]['clear_RMSE'].tolist();out[name]['true_weak_RMSE']=out[name]['true_weak_RMSE'].tolist()
    out['GPU']=gpu;(A/'weak_confirmation.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
