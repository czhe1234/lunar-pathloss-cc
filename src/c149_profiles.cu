// C149 audit-only exact registered point profiles. No model, labels, or masks.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
struct Best { double highest, nu, at; int pieces; };
__device__ void point(Best &v, double t, double a, double b, double scale) {
    double e=a+b*t;
    v.highest=fmax(v.highest,e);
    if(t>0.0 && t<1.0) {
        double n=e*scale/sqrt(t*(1.0-t));
        if(n>v.nu){v.nu=n;v.at=t;}
    }
}
__device__ void segment(Best &v,double lo,double hi,double a,double b,double scale) {
    if(hi<=lo)return;
    ++v.pieces;point(v,lo,a,b,scale);point(v,hi,a,b,scale);
    double den=b+2*a;
    if(fabs(den)>1e-20) {
        double t=a/den;
        if(t>lo && t<hi)point(v,t,a,b,scale);
    }
}
__device__ double height(const float *h,double x,double y) {
    int ix=min(254,max(0,int(floor(x)))),iy=min(254,max(0,int(floor(y))));
    double fx=x-ix,fy=y-iy;
    double a=h[iy*256+ix],b=h[iy*256+ix+1],c=h[(iy+1)*256+ix],d=h[(iy+1)*256+ix+1];
    return fy>fx ? a+(d-c)*fx+(c-a)*fy : a+(b-a)*fx+(d-b)*fy;
}
__global__ void profiles(const float *h,int tr,int tc,double *out) {
    int pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel>=65536)return;
    int r=pixel/256,c=pixel%256;
    double x=tc*(255.0/256.0),y=(tr+1)*(255.0/256.0);
    double z=floor(double(h[min(tr+1,255)*256+tc]))+3.0;
    double rx=1.0/3.0+c*((255.0-2.0/3.0)/255.0);
    double ry=1.0/3.0+r*((255.0-2.0/3.0)/255.0);
    double rz=height(h,rx,ry)+1.0,dx=rx-x,dy=ry-y,dz=rz-z;
    double distance=(256.0/255.0)*hypot(dx,dy),e0=height(h,x,y)-z;
    int flags=0;if(distance<1e-10)flags|=1;if(e0>0)flags|=2;if(e0==0)flags|=4;
    Best v;v.highest=-CUDART_INF;v.nu=-CUDART_INF;v.at=0;v.pieces=0;
    double scale=distance>1e-10?sqrt(2.0/distance):0.0;
    int ix=min(254,max(0,int(floor(x+1e-9*dx)))),iy=min(254,max(0,int(floor(y+1e-9*dy))));
    int xs=dx>0?1:-1,ys=dy>0?1:-1;
    double sx=fabs(dx)>1e-16?fabs(1.0/dx):1e100,sy=fabs(dy)>1e-16?fabs(1.0/dy):1e100;
    double nx=sx<1e99?(ix+(xs>0)-x)/dx:1e100,ny=sy<1e99?(iy+(ys>0)-y)/dy:1e100;
    double enter=0;bool done=false;
    for(int it=0;it<515 && ix>=0 && ix<255 && iy>=0 && iy<255;++it) {
        double end=fmin(1.0,fmin(nx,ny));
        double td=fabs(dx-dy)>1e-16?((y-iy)-(x-ix))/(dx-dy):2.0;
        double split=(td>enter+1e-12 && td<end-1e-12)?td:end,lo=enter;
        for(int k=0;k<2;++k) {
            double hi=k==0?split:end;
            if(hi>lo+1e-14) {
                double mid=.5*(lo+hi),fx=x-ix+dx*mid,fy=y-iy+dy*mid;
                double a=h[iy*256+ix],b=h[iy*256+ix+1],c0=h[(iy+1)*256+ix],d=h[(iy+1)*256+ix+1];
                double gx=fy>fx?d-c0:b-a,gy=fy>fx?c0-a:d-b;
                double off=a+gx*(x-ix)+gy*(y-iy)-z,slope=gx*dx+gy*dy-dz;
                segment(v,lo,hi,off,slope,scale);
            }
            lo=hi;if(lo>=end)break;
        }
        if(end>=1.0-1e-13){done=true;break;}
        bool ax=nx<=ny,ay=ny<=nx;
        if(ax){ix+=xs;nx+=sx;}if(ay){iy+=ys;ny+=sy;}enter=end;
    }
    if(!done)flags|=8;
    if(e0>0){v.nu=CUDART_INF;v.at=0;}else if(e0==0 && v.nu<0){v.nu=0;v.at=0;}
    if(distance<1e-10){v.nu=0;v.at=0;}
    // planes: exact height excess, nu(lambda=1m), t_argmax, source excess, flags.
    out[pixel]=v.highest;out[65536+pixel]=v.nu;out[2*65536+pixel]=v.at;
    out[3*65536+pixel]=e0;out[4*65536+pixel]=flags;
}
extern "C" int c149_profiles(const float *h,int tr,int tc,double *out,cudaStream_t stream) {
    profiles<<<512,128,0,stream>>>(h,tr,tc,out);return cudaGetLastError();
}
