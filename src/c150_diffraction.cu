// C150: deterministic registered piecewise-planar diffraction descriptors.
// No labels, validity masks, learned predictors, or Monte-Carlo solver calls.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
struct Summary {
    double highest, nu, at, positive_integral, slope_tx, slope_rx;
    bool done;
};
__device__ double terrain(const float *h,double x,double y) {
    int ix=min(254,max(0,int(floor(x)))),iy=min(254,max(0,int(floor(y))));
    double fx=x-ix,fy=y-iy;
    double a=h[iy*256+ix],b=h[iy*256+ix+1],c=h[(iy+1)*256+ix],d=h[(iy+1)*256+ix+1];
    return fy>fx ? a+(d-c)*fx+(c-a)*fy : a+(b-a)*fx+(d-b)*fy;
}
__device__ double primitive(double t,double a,double b) {
    return (2*a+b)*asin(sqrt(fmax(0.0,fmin(1.0,t))))-b*sqrt(fmax(0.0,t*(1-t)));
}
__device__ void point(Summary &v,double t,double a,double b,double dz,double distance,double scale,bool full) {
    double e=a+b*t;
    v.highest=fmax(v.highest,e);
    if(t>0 && t<1) {
        double n=e*scale/sqrt(t*(1-t));
        if(n>v.nu){v.nu=n;v.at=t;}
        if(full) {
            v.slope_tx=fmax(v.slope_tx,(e+dz*t)/(t*distance));
            v.slope_rx=fmax(v.slope_rx,(e-dz*(1-t))/((1-t)*distance));
        }
    }
}
__device__ void piece(Summary &v,double lo,double hi,double a,double b,double dz,double distance,double scale,bool full) {
    if(hi<=lo)return;
    point(v,lo,a,b,dz,distance,scale,full);point(v,hi,a,b,dz,distance,scale,full);
    double den=b+2*a;
    if(fabs(den)>1e-20) {
        double critical=a/den;
        if(critical>lo && critical<hi)point(v,critical,a,b,dz,distance,scale,full);
    }
    if(full) {
        double e0=a+b*lo,e1=a+b*hi,left=lo,right=hi;
        if(e0<=0 && e1<=0)return;
        if((e0>0)!=(e1>0)) {
            double zero=-a/b;
            if(e0<=0)left=zero;else right=zero;
        }
        v.positive_integral += fmax(0.0,(primitive(right,a,b)-primitive(left,a,b))*scale);
    }
}
__device__ Summary trace(const float *h,double x,double y,double z,double rx,double ry,double rz,bool touch_tx,bool touch_rx,bool full) {
    double dx=rx-x,dy=ry-y,dz=rz-z,distance=(256.0/255.0)*hypot(dx,dy);
    Summary v;v.highest=-CUDART_INF;v.nu=(touch_tx||touch_rx)?0:-CUDART_INF;
    v.at=0;v.positive_integral=0;v.slope_tx=-CUDART_INF;v.slope_rx=-CUDART_INF;v.done=false;
    if(distance<=1e-10)return v;
    double scale=sqrt(2.0/distance);
    int ix=min(254,max(0,int(floor(x+1e-9*dx)))),iy=min(254,max(0,int(floor(y+1e-9*dy))));
    int xs=dx>0?1:-1,ys=dy>0?1:-1;
    double sx=fabs(dx)>1e-16?fabs(1.0/dx):1e100,sy=fabs(dy)>1e-16?fabs(1.0/dy):1e100;
    double nx=sx<1e99?(ix+(xs>0)-x)/dx:1e100,ny=sy<1e99?(iy+(ys>0)-y)/dy:1e100;
    double enter=0;
    for(int it=0;it<515 && ix>=0 && ix<255 && iy>=0 && iy<255;++it) {
        double end=fmin(1.0,fmin(nx,ny));
        double td=fabs(dx-dy)>1e-16?((y-iy)-(x-ix))/(dx-dy):2;
        double split=(td>enter+1e-12 && td<end-1e-12)?td:end,lo=enter;
        for(int k=0;k<2;++k) {
            double hi=k==0?split:end;
            if(hi>lo+1e-14) {
                double mid=.5*(lo+hi),fx=x-ix+dx*mid,fy=y-iy+dy*mid;
                double a=h[iy*256+ix],b=h[iy*256+ix+1],c=h[(iy+1)*256+ix],d=h[(iy+1)*256+ix+1];
                double gx=fy>fx?d-c:b-a,gy=fy>fx?c-a:d-b;
                double off=a+gx*(x-ix)+gy*(y-iy)-z,slope=gx*dx+gy*dy-dz;
                // An obstacle endpoint is exactly on terrain. Do not amplify cancellation into a false extra knife edge.
                if(touch_tx && lo<=1e-13)off=0;
                if(touch_rx && hi>=1-1e-13)slope=-off;
                piece(v,lo,hi,off,slope,dz,distance,scale,full);
            }
            lo=hi;if(lo>=end)break;
        }
        if(end>=1-1e-13){v.done=true;break;}
        bool ax=nx<=ny,ay=ny<=nx;
        if(ax){ix+=xs;nx+=sx;}if(ay){iy+=ys;ny+=sy;}enter=end;
    }
    return v;
}
__device__ double loss_j(double nu) {
    if(nu<=-.78)return 0;
    double u=nu-.1;
    return fmax(0.0,6.9+20*log10(sqrt(u*u+1)+u));
}
__global__ void features(const float *h,int tr,int tc,const float *reuse,float *out) {
    int pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel>=65536)return;
    int row=pixel/256,col=pixel%256;
    double x=tc*(255.0/256.0),y=(tr+1)*(255.0/256.0);
    double z=floor(double(h[min(tr+1,255)*256+tc]))+3;
    double rx=1.0/3.0+col*((255.0-2.0/3.0)/255.0),ry=1.0/3.0+row*((255.0-2.0/3.0)/255.0);
    double rz=terrain(h,rx,ry)+1,dx=rx-x,dy=ry-y,dz=rz-z;
    double distance=(256.0/255.0)*hypot(dx,dy),d3=hypot(distance,dz);
    Summary root=trace(h,x,y,z,rx,ry,rz,false,false,true);
    bool ok=root.done && terrain(h,x,y)<z && distance>1e-10 && isfinite(root.nu) && isfinite(root.at);
    double left_nu=0,right_nu=0;
    if(ok && (root.nu/sqrt(299792458.0/415e6)>-.78 || root.nu/sqrt(299792458.0/5.775e9)>-.78)) {
        // Classical three-edge approximation: one primary and at most one genuinely blocking edge on either side.
        // Internal primary position remains Float64; FP32 C149 argmax must never define secondary endpoint geometry.
        double mx=x+root.at*dx,my=y+root.at*dy,mz=terrain(h,mx,my);
        if(root.at>1e-12) {
            Summary left=trace(h,x,y,z,mx,my,mz,false,true,false);
            ok=ok&&left.done;left_nu=left.nu;
        }
        if(root.at<1-1e-12) {
            Summary right=trace(h,mx,my,mz,rx,ry,rz,true,false,false);
            ok=ok&&right.done;right_nu=right.nu;
        }
    }
    double reported_nu=reuse?double(reuse[pixel]):double(float(root.nu));
    double reported_at=reuse?double(reuse[65536+pixel]):double(float(root.at));
    for(int band=0;band<2;++band) {
        double frequency=band==0?415e6:5.775e9,lambda=299792458.0/frequency;
        double nu=root.nu/sqrt(lambda),deygout=loss_j(nu);
        if(deygout>0) {
            double nl=left_nu/sqrt(lambda),nr=right_nu/sqrt(lambda);
            if(nl>1e-9)deygout+=loss_j(nl);
            if(nr>1e-9)deygout+=loss_j(nr);
        }
        double bull_nu=nu;
        if(root.highest>1e-10) {
            double den=root.slope_tx+root.slope_rx;
            double db=(rz-z+root.slope_rx*distance)/den;
            if(den<=0 || db<=0 || db>=distance)ok=false;
            else {
                double excess=(root.slope_tx-dz/distance)*db;
                bull_nu=excess*sqrt(2*distance/(lambda*db*(distance-db)));
            }
        }
        double luc=loss_j(bull_nu),bull=luc+(1-exp(-luc/6))*(10+.02*distance/1000);
        int offset=band*5;
        out[(offset+0)*65536+pixel]=ok?float(reported_nu/sqrt(lambda)):CUDART_NAN_F;
        out[(offset+1)*65536+pixel]=ok?float(reported_at):CUDART_NAN_F;
        out[(offset+2)*65536+pixel]=ok?float(root.positive_integral/sqrt(lambda)):CUDART_NAN_F;
        out[(offset+3)*65536+pixel]=ok?float(deygout):CUDART_NAN_F;
        out[(offset+4)*65536+pixel]=ok?float(bull):CUDART_NAN_F;
        out[(10+band)*65536+pixel]=ok?float(20*log10(4*3.14159265358979323846*d3/lambda)):CUDART_NAN_F;
        if(band==0){out[12*65536+pixel]=float(root.nu);out[13*65536+pixel]=float(root.at);}
    }
}
extern "C" int c150_diffraction(const float *h,int tr,int tc,const float *reuse,float *out,cudaStream_t stream) {
    features<<<512,128,0,stream>>>(h,tr,tc,reuse,out);return cudaGetLastError();
}
