"""Exact finite-input LightGBM regression trees on the authorized CUDA device.
Thresholds and leaf values remain FP64. No training or approximate tree fitting.
"""
import numpy as np
import torch,triton
import triton.language as tl
@triton.jit
def _trees(X,ROOT,FID,LEFT,RIGHT,THRESH,VALUE,OUT,N:tl.constexpr,T:tl.constexpr,D:tl.constexpr,B:tl.constexpr):
 z=tl.program_id(0)*B+tl.arange(0,B);keep=z<N*T;p=z%N;t=z//N
 node=tl.load(ROOT+t,keep,other=0)
 for depth in range(D):
  f=tl.load(FID+node,keep,other=-1);active=keep&(f>=0)
  v=tl.load(X+tl.maximum(f,0)*N+p,active,other=0).to(tl.float64)
  th=tl.load(THRESH+node,active,other=0)
  le=tl.load(LEFT+node,active,other=0);ri=tl.load(RIGHT+node,active,other=0)
  node=tl.where(active,tl.where(v<=th,le,ri),node)
 val=tl.load(VALUE+node,keep,other=0)
 tl.store(OUT+z,val,keep)

class Forest:
 def __init__(self,model):
  dump=model.dump_model();assert dump['objective']=='regression'
  roots=[];fid=[];left=[];right=[];threshold=[];values=[];maxdepth=0
  def visit(node,depth):
   nonlocal maxdepth
   maxdepth=max(maxdepth,depth);i=len(fid);fid.append(-1);left.append(i);right.append(i);threshold.append(0.);values.append(0.)
   if 'leaf_value' in node:values[i]=float(node['leaf_value'])
   else:
    assert node['decision_type']=='<=' and node['missing_type'] in ['None','NaN']
    fid[i]=int(node['split_feature']);threshold[i]=float(node['threshold']);left[i]=visit(node['left_child'],depth+1);right[i]=visit(node['right_child'],depth+1)
   return i
  for tree in dump['tree_info']:roots.append(visit(tree['tree_structure'],0))
  tensor=lambda a,d:torch.tensor(a,dtype=d,device='cuda')
  self.arrays=[tensor(roots,torch.int32),tensor(fid,torch.int32),tensor(left,torch.int32),tensor(right,torch.int32),tensor(threshold,torch.float64),tensor(values,torch.float64)]
  self.ntrees=len(roots);self.depth=maxdepth;self.nfeatures=model.num_feature();self.nodes=len(fid)
 @torch.inference_mode()
 def predict(self,x):
  assert x.dtype==torch.float32 and x.is_cuda and x.ndim==2 and x.shape[1]==self.nfeatures
  x=x.T.contiguous();n=x.shape[1]
  out=torch.empty((self.ntrees,n),dtype=torch.float64,device='cuda')
  _trees[(triton.cdiv(self.ntrees*n,256),)](x,*self.arrays,out,n,self.ntrees,self.depth,256,num_warps=4)
  return out.sum(0)

