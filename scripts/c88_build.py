"""Extend the exact C70 ray walk to true variable-resolution, per-cell topology."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
s=(R/'src/c70_geometry.cu').read_text()
def subst(old,new):
    global s
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new)
subst('constexpr int C=255*255;\nconstexpr int N=2*C;\nconstexpr int K=27;','constexpr int K=29;')
subst('int n;\n};','int n; float transmit[2];\n};')
subst('float spacing,float distance) {','float spacing,float distance,const float2 *slab) {')
subst('if(!v.n)v.first=cosn;', '''float ct=fminf(1.f,fmaxf(0.f,cosn))*4096.f;int jj=min(4095,int(ct)); float mix=ct-jj;
        float2 a=slab[jj],b=slab[jj+1];v.transmit[0]+=(1-mix)*a.x+mix*b.x;v.transmit[1]+=(1-mix)*a.y+mix*b.y;
        if(!v.n)v.first=cosn;''')
subst('float rz,float s,int anti) {','float rz,float s,const unsigned char *diags,int m,const float2 *slab) {')
subst('v.mincos=1;','v.mincos=1;v.transmit[0]=v.transmit[1]=0;')
s=s.replace('min(254,max(0,','min(m-1,max(0,')
subst('for(int it=0;it<515 && ix>=0 && ix<255 && iy>=0 && iy<255;++it){','for(int it=0;it<2*m+5 && ix>=0 && ix<m && iy>=0 && iy<m;++it){\n        int anti=diags[iy*m+ix];')
subst('piece(v,g[iy*255+ix+part*C],a,b,x,y,ix,iy,dx,dy,z,dz,s,fmaxf(.1,dist));','piece(v,g[iy*m+ix+part*m*m],a,b,x,y,ix,iy,dx,dy,z,dz,s,fmaxf(.1,dist),slab);')
subst('int anti,float *out) {\n    int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=N)return;\n    int cell=face%C,ix=cell%255,iy=cell/255;', '''const unsigned char *diags,int m,const int *ids,int n,const float2 *slab,float *out) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=n)return;int face=ids[id],C=m*m;
    int cell=face%C,ix=cell%m,iy=cell/m,anti=diags[cell];''')
subst('spacing,anti);','spacing,diags,m,slab);')
subst('mn=fminf(v.required,mn);','acc[27]+=powf(10.f,-fminf(110.f,v.transmit[0])*.1f);acc[28]+=powf(10.f,-fminf(110.f,v.transmit[1])*.1f);\n        mn=fminf(v.required,mn);')
subst('for(int k=0;k<K-2;++k)out[face*K+k]=acc[k]/nq;','for(int k=0;k<K;++k)out[id*K+k]=acc[k]/nq;')
s=s.replace('out[face*K+','out[id*K+')
subst('int ground_geometry(','int variable_geometry(')
subst('int anti,float *output,cudaStream_t stream){\n    kernel<<<(N+127)/128,128,0,stream>>>(g,quadrature,nq,x,y,z,s,anti,output);','const unsigned char *diags,int m,const int *ids,int n,const float2 *slab,float *output,cudaStream_t stream){\n    kernel<<<(n+127)/128,128,0,stream>>>(g,quadrature,nq,x,y,z,s,diags,m,ids,n,slab,output);')
(R/'src/c88_geometry.cu').write_text(s)
