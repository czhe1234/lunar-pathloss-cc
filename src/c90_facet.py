"""Learn receiver microfacet orientations from terrain only, then evaluate their directional response."""
import argparse,csv,json,time,sys,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
A=R/'audit/c90_facet'
class Facets(nn.Module):
    def __init__(self,arm):
        super().__init__();self.arm=arm
        self.terrain=nn.Sequential(nn.Linear(49,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,32))
        self.patch=nn.Sequential(nn.Linear(49,64),nn.SiLU(),nn.Linear(64,32))
        self.input=nn.Linear(45+2+2+4+8+32+8*5,128)
        self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(128),nn.Linear(128,192),nn.SiLU(),nn.Linear(192,128)) for _ in range(3)])
        self.head=nn.Linear(128,4);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        h=x[:,:49];comp=x[:,49:94];p=x[:,94:96];fs=x[:,96:98];ray=x[:,98:101]
        terrain=self.terrain(h).float().reshape(-1,8,4);g=h.reshape(-1,7,7)
        # At the output cell keep directional one-sided slopes, avoiding a single averaged normal.
        xp=g[:,3,4]-g[:,3,3];xm=g[:,3,3]-g[:,3,2];yp=g[:,4,3]-g[:,3,3];ym=g[:,3,3]-g[:,2,3]
        gx=torch.stack([xp,xp,xm,xm,xp,xp,xm,xm],1);gy=torch.stack([yp,ym,yp,ym,yp,ym,yp,ym],1)
        scale=.025+.55*torch.maximum((xp-xm).abs(),(yp-ym).abs()).clamp(max=1)
        delta=torch.tanh(terrain[:,:,:2])*scale[:,None,None]
        if self.arm!='fixed':gx=gx+delta[:,:,0];gy=gy+delta[:,:,1]
        width=.0005+.07*torch.sigmoid(terrain[:,:,2]);weight=torch.softmax(terrain[:,:,3],1)
        xy=x[:,102:104] if self.arm=='rotated' else ray[:,:2]
        cos=(ray[:,2:3]-xy[:,0:1]*gx-xy[:,1:2]*gy)/(1+gx.square()+gy.square()).sqrt()
        u=(cos/width).clamp(-25,25);lobe=1/(1+u.square())
        feat=torch.cat([weight*lobe,weight*torch.exp(-.5*u.square()),torch.tanh(u),torch.log1p(u.abs()),weight],1)
        z=self.input(torch.cat([comp,(p-fs)/15,(fs-80)/30,ray,x[:,101:102]/6,x[:,104:112],self.patch(h),feat],1))
        for block in self.blocks:z=z+.35*block(z)
        out=self.head(z).float();common=out[:,:1]*4;delta=out[:,1:2]*3
        return p+torch.cat([common-delta,common+delta],1),out[:,2:]
class Data:
    def __init__(self,group,device):
        paths=sorted(A.glob(group+'[0-9]'));assert len(paths)==3
        metas=[json.load(open(d/'done.json')) for d in paths]
        self.names=np.concatenate([m['terrains'] for m in metas]);self.ids=np.concatenate([m['ids'] for m in metas]);self.m=metas[0]['pixels']
        assert len(set(self.ids))==len(self.ids)
        self.x=torch.from_numpy(np.concatenate([np.load(d/'x.npy').reshape(-1,112) for d in paths])).to(device)
        self.t=torch.from_numpy(np.concatenate([np.load(d/'targets.npy').reshape(-1,6) for d in paths])).to(device)
    def get(self,q):
        if self.x.device.type=='cpu' and torch.is_tensor(q):q=q.cpu()
        return self.x[q].cuda().float(),self.t[q].cuda().float()
def sse(pred,y,w):
    return ((pred.double()-y.double()).square()*w).sum(1).cpu().numpy()
@torch.inference_mode()
def predict(net,D):
    net.eval();v=[]
    for j in range(0,len(D.x),16384):
        x,t=D.get(slice(j,j+16384))
        with torch.autocast('cuda',torch.bfloat16):a,_=net(x)
        v.append(a.cpu().numpy())
    return np.concatenate(v)
def evaluate(D,a,beta=None):
    y=D.t[:,:2].cpu().numpy();w=D.t[:,2:4].cpu().numpy().astype(bool);p=D.t[:,4:6].cpu().numpy()
    vis=D.x[:,65].cpu().numpy().astype('float32');fs=D.x[:,96:98].cpu().numpy().astype('float32')
    zh=(vis<.01).astype(int)*2+((vis>=.01)&(vis<=.99));state=2*zh[:,None]+((p-fs)>3)
    d=(a-p).clip(-30,30)
    if beta is None:
        beta=np.zeros((2,6))
        for b in range(2):
            for s in range(6):
                k=(state[:,b]==s)&w[:,b];dx=d[k,b].astype('float64');dy=(y[k,b]-p[k,b]).astype('float64')
                beta[b,s]=(np.sum(dx*dy)/(np.sum(dx**2)+1)).clip(0,1)
    cand=p+np.stack([beta[b,state[:,b]]*d[:,b] for b in range(2)],1)
    t=[]
    for pred in (p,a,cand):
        e=((pred.astype('float64')-y)**2*w).reshape(-1,D.m,2).sum(1);t.append(e)
    nw=w.reshape(-1,D.m,2).sum(1);t=np.stack(t,1)
    return beta,t,nw,np.sqrt(t.sum(0)/nw.sum(0))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True,choices=['real','rotated','fixed'])
    ap.add_argument('--confirm',action='store_true');ap.add_argument('--epochs',type=int,default=8);args=ap.parse_args()
    gpu=gpu_guard();torch.set_num_threads(3);torch.manual_seed(90607);np.random.seed(90607)
    torch.set_float32_matmul_precision('high')
    run=R/'runs'/('c90_'+args.arm);run.mkdir(exist_ok=True)
    net=Facets(args.arm).cuda();ts=time.monotonic()
    if args.confirm:
        ck=torch.load(run/'best.pt',weights_only=True,map_location='cpu');net.load_state_dict(ck['state'])
        D=Data('confirm','cpu');y=predict(net,D);beta=np.array(ck['beta'])
        _,v,n,r=evaluate(D,y,beta);gain=r[0]-r[2]
        np.savez(run/'paired.npz',sse=v,count=n,terrain=D.names,ids=D.ids)
        atomic_json(run/'confirmation.json',dict(arm=args.arm,maps=len(D.ids),terrains=len(set(D.names)),rmse=r.tolist(),
                   gain=gain.tolist(),gpu=gpu,epoch=ck['epoch'],weights=ck['beta'],scope='2048 uniform official TRAIN positions/map, not official VAL'))
        print('confirm',args.arm,r.round(5),gain.round(5),flush=True);return
    resident=torch.cuda.mem_get_info()[0]>(4*2**30);device='cuda' if resident else 'cpu'
    D=Data('fit',device);cal=Data('cal','cpu');sel=Data('select','cpu')
    for d in (cal,sel):assert set(D.names).isdisjoint(d.names)
    assert set(cal.names).isdisjoint(sel.names)
    atomic_json(run/'config.json',dict(gpu=gpu,arm=args.arm,fit_maps=len(D.ids),fit_terrains=len(set(D.names)),fit_points=len(D.x),
        params=sum(p.numel() for p in net.parameters()),batch=8192,epochs=args.epochs,resident=resident,seed=90607))
    opt=torch.optim.AdamW(net.parameters(),lr=.00115,weight_decay=.012,fused=True);gen=torch.Generator().manual_seed(904321)
    best=float('inf');history=[];print('train',args.arm,len(D.x),'points',resident,flush=True)
    for ep in range(1,args.epochs+1):
        net.train();order=torch.randperm(len(D.x),generator=gen,device='cpu')
        for param in opt.param_groups:param['lr']=.00115*(.2+.8*.5*(1+math.cos((ep-1)*math.pi/args.epochs)))
        loss_tot=torch.zeros((),device='cuda');steps=0
        for st in range(0,len(order),8192):
            ix=order[st:st+8192].to(device);x,t=D.get(ix);y=t[:,:2];w=t[:,2:4]
            with torch.autocast('cuda',torch.bfloat16):a,logit=net(x)
            err=(a-y).square()
            fit=(err*w).sum(0)/w.sum(0).clamp_min(1)
            # Auxiliary clear-weak event per band. All regression terms use official-valid targets only.
            visible=(x[:,65:66]>.99);weak=(y-x[:,96:98]>3).float()
            aux=F.binary_cross_entropy_with_logits(logit,weak,reduction='none')
            aw=w*visible*(1+3*weak)
            joint=w[:,0]*w[:,1]*(x[:,65]<.01)
            dual=(((a[:,1]-a[:,0])-(y[:,1]-y[:,0])).square()*joint).sum()/joint.sum().clamp_min(1)
            loss=.5*fit.sum()+.06*dual+.5*(aux*aw).sum()/aw.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),12);opt.step()
            loss_tot+=loss.detach();steps+=1
        beta,_,_,rc=evaluate(cal,predict(net,cal));_,_,_,rs=evaluate(sel,predict(net,sel),beta)
        item=dict(epoch=ep,train_loss=float(loss_tot/steps),selection=rs.tolist(),cal=rc.tolist(),
            gain=(rs[0]-rs[2]).tolist(),seconds=round(time.monotonic()-ts,1))
        history.append(item);atomic_json(run/'history.json',history)
        print('epoch',ep,'loss',round(item['train_loss'],4),'select standalone',rs[1].round(4),'C82-blend gain',np.round(item['gain'],4),'seconds',item['seconds'],flush=True)
        if rs[2].mean()<best:
            best=float(rs[2].mean());torch.save(dict(state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                 beta=beta.tolist(),epoch=ep,selection=rs.tolist()),run/'best.pt')
    atomic_json(run/'done.json',dict(epochs=args.epochs,seconds=time.monotonic()-ts,selection='48 TRAIN terrain score; no official VAL used'))
if __name__=='__main__':main()
