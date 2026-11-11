// Geometric quantities only. Piecewise-planar terrain, exact extrema at triangle boundaries.
#include <cuda_runtime.h>
#include <cmath>
constexpr int C=255*255;
constexpr int N=2*C;
constexpr int K=27;
struct Summary {
    float bins[8],highest,required,where,under,area,first,last,mincos;
    int n;
};
__device__ void endpoint(Summary &v,float t,float e) {
    if(t>1e-5f && t<.99999f)v.required=fmaxf(v.required,e/(1-t));
    if(e>v.highest){v.highest=e;v.where=t;}
    int bin=min(7,int(t*8));v.bins[bin]=fmaxf(v.bins[bin],e);
}
__device__ void piece(Summary &v,const float4 g,float ta,float tb,float x,float y,
        int ix,int iy,float dx,float dy,float z,float dz,float spacing,float distance) {
    if(tb-ta<1e-7f)return;
    float off=g.x+g.y*(x-ix)+g.z*(y-iy)-z;
    float slope=g.y*dx+g.z*dy-dz;
    float e0=fmaf(ta,slope,off),e1=fmaf(tb,slope,off);
    endpoint(v,ta,e0);endpoint(v,tb,e1);
    // A short path can cross radial interval boundaries inside one terrain triangle.
    int k0=min(7,int(ta*8)),k1=min(7,int(tb*8));
    for(int k=k0+1;k<=k1;++k){
        float at=fmaf(k*.125f,slope,off);
        v.bins[k-1]=fmaxf(v.bins[k-1],at);v.bins[k]=fmaxf(v.bins[k],at);
    }
    float dt=tb-ta;
    if(e0>=0 && e1>=0){v.under+=dt;v.area+=dt*.5f*(e0+e1);}
    else if((e0>0)!=(e1>0)) {
        float root=ta-dt*e0/(e1-e0);
        float length=e0>0?root-ta:tb-root;
        v.under+=length;v.area+=.5f*length*fmaxf(e0,e1);
        float cosn=fabsf(slope)/(distance*sqrtf(1+(g.y*g.y+g.z*g.z)/(spacing*spacing)));
        if(!v.n)v.first=cosn;
        v.last=cosn;v.mincos=fminf(v.mincos,cosn);++v.n;
    }
}
__device__ Summary trace(const float4 *g,float x,float y,float z,float rx,float ry,float rz,float s,int anti) {
    const float low=-1000;
    Summary v;for(int k=0;k<8;++k)v.bins[k]=low;
    v.highest=low;v.required=low;v.where=0;v.under=0;v.area=0;v.n=0;v.first=0;v.last=0;v.mincos=1;
    float dx=rx-x,dy=ry-y,dz=rz-z;
    float dist=sqrtf((dx*dx+dy*dy)*s*s+dz*dz);
    int ix=min(254,max(0,int(floorf(x+1e-6f*dx)))),iy=min(254,max(0,int(floorf(y+1e-6f*dy))));
    int xs=dx>0?1:-1,ys=dy>0?1:-1;
    float sx=fabsf(dx)>1e-7f?fabsf(1/dx):1e20f,sy=fabsf(dy)>1e-7f?fabsf(1/dy):1e20f;
    float nx=sx<1e19f?(ix+(xs>0)-x)/dx:1e20f,ny=sy<1e19f?(iy+(ys>0)-y)/dy:1e20f;
    float enter=0;
    for(int it=0;it<515 && ix>=0 && ix<255 && iy>=0 && iy<255;++it){
        float end=fminf(1.f,fminf(nx,ny));
        float dd=anti?dx+dy:dx-dy;
        float tdiag=fabsf(dd)>1e-8f?(anti?1-(x-ix)-(y-iy):(y-iy)-(x-ix))/dd:2.f;
        float split=(tdiag>enter+1e-7f && tdiag<end-1e-7f)?tdiag:end;
        float a=enter;
        for(int i=0;i<2;++i){
            float b=i==0?split:end;
            if(b>a+1e-7f){
                float mid=.5f*(a+b),fx=x-ix+dx*mid,fy=y-iy+dy*mid;
                int part=anti?(fx+fy>1):(fy>fx);
                piece(v,g[iy*255+ix+part*C],a,b,x,y,ix,iy,dx,dy,z,dz,s,fmaxf(.1,dist));
            }
            a=b;if(a>=end)break;
        }
        if(end>=1.f)break;
        bool ax=nx<=ny,ay=ny<=nx;
        if(ax){ix+=xs;nx+=sx;}if(ay){iy+=ys;ny+=sy;}enter=end;
    }
    return v;
}
__global__ void kernel(const float4 *g,const float3 *quad,int nq,float x,float y,float z,float spacing,int anti,float *out) {
    int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=N)return;
    int cell=face%C,ix=cell%255,iy=cell/255;float4 p=g[face];
    float acc[K]={0};float mn=1e6f,mx=-1e6f;
    for(int j=0;j<nq;++j){
        auto b=quad[j];float fx,fy;
        if(!anti && face<C){fx=b.y+b.z;fy=b.z;}
        else if(!anti){fx=b.y;fy=b.y+b.z;}
        else if(face<C){fx=b.y;fy=b.z;}else{fx=b.x+b.z;fy=b.x+b.y;}
        float rz=p.x+p.y*fx+p.z*fy+1;
        float dx=ix+fx-x,dy=iy+fy-y,dz=rz-z;
        float d=sqrtf((dx*dx+dy*dy)*spacing*spacing+dz*dz);
        auto v=trace(g,x,y,z,ix+fx,iy+fy,rz,spacing,anti);
        for(int k=0;k<8;++k)acc[k]+=asinhf(v.bins[k]/1.5f)/3;
        float cosg=(dz-p.y*dx-p.z*dy)/(fmaxf(.25,d)*sqrtf(1+(p.y*p.y+p.z*p.z)/(spacing*spacing)));
        acc[8]+=asinhf(v.highest/.25f)/3;
        acc[9]+=asinhf(v.required/.5f)/3;
        acc[10]+=v.where;acc[11]+=v.under;acc[12]+=log1pf(v.under*d)/4;acc[13]+=log1pf(v.area*d)/4;
        acc[14]+=log1pf(v.n)/2;acc[15]+=v.first;acc[16]+=v.last;acc[17]+=v.n?v.mincos:0;
        acc[18]+=(v.highest<0);acc[19]+=cosg;acc[20]+=log1pf(4e9f/(4*3.141593f)*p.w*fabsf(cosg)/fmaxf(.0625,d*d))/10;
        acc[21]+=(v.required<-.3f);acc[22]+=(v.required<.3f);acc[23]+=p.y/spacing;acc[24]+=p.z/spacing;
        mn=fminf(v.required,mn);mx=fmaxf(v.required,mx);
    }
    for(int k=0;k<K-2;++k)out[face*K+k]=acc[k]/nq;
    out[face*K+25]=asinhf(mn/.5f)/3;out[face*K+26]=asinhf(mx/.5f)/3;
}
extern "C" int ground_geometry(const float4 *g,const float3 *quadrature,int nq,float x,float y,float z,float s,int anti,float *output,cudaStream_t stream){
    kernel<<<(N+127)/128,128,0,stream>>>(g,quadrature,nq,x,y,z,s,anti,output);
    return cudaGetLastError();
}
