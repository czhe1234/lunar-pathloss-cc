// C156 CPU-only registered point profile crossing counts; no power solver.
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
struct RayStat { int crossings=0,blocked=0,done=0; double highest=-1e100,under=0; bool active=false; };
static double height(const float*h,double x,double y){
 int ix=std::clamp(int(std::floor(x)),0,254),iy=std::clamp(int(std::floor(y)),0,254);
 double fx=x-ix,fy=y-iy,a=h[iy*256+ix],b=h[iy*256+ix+1],c=h[(iy+1)*256+ix],d=h[(iy+1)*256+ix+1];
 return fy>fx?a+(d-c)*fx+(c-a)*fy:a+(b-a)*fx+(d-b)*fy;
}
static void piece(RayStat&s,double lo,double hi,double a,double b){
 if(hi<=lo+1e-14)return;
 double e0=a+b*lo,e1=a+b*hi;s.highest=std::max({s.highest,e0,e1});
 // Positive intervals, with a1e-10m numerical zero tolerance.
 constexpr double eps=1e-10;
 bool pos0=e0>eps,pos1=e1>eps;
 if(pos0!=s.active){++s.crossings;if(pos0)++s.blocked;s.active=pos0;}
 if(pos0!=pos1){++s.crossings;if(pos1)++s.blocked;s.active=pos1;}
 if(pos0 && pos1)s.under+=hi-lo;
 else if(pos0!=pos1){double root=lo-(hi-lo)*e0/(e1-e0);s.under+=pos0?root-lo:hi-root;}
}
static RayStat trace(const float*h,int tr,int tc,int pix){
 int r=pix/256,c=pix%256;double x=tc*(255./256),y=(tr+1)*(255./256);
 double z=std::floor(double(h[std::min(tr+1,255)*256+tc]))+3;
 double rx=1./3+c*((255.-2./3)/255),ry=1./3+r*((255.-2./3)/255);
 double rz=height(h,rx,ry)+1,dx=rx-x,dy=ry-y,dz=rz-z;
 double e0=height(h,x,y)-z;RayStat s;s.active=e0>1e-10;s.blocked=s.active;
 int ix=std::clamp(int(std::floor(x+1e-9*dx)),0,254),iy=std::clamp(int(std::floor(y+1e-9*dy)),0,254);
 int xs=dx>0?1:-1,ys=dy>0?1:-1;
 double sx=std::abs(dx)>1e-16?std::abs(1/dx):1e100,sy=std::abs(dy)>1e-16?std::abs(1/dy):1e100;
 double nx=sx<1e99?(ix+(xs>0)-x)/dx:1e100,ny=sy<1e99?(iy+(ys>0)-y)/dy:1e100,enter=0;
 for(int it=0;it<515 && ix>=0 && ix<255 && iy>=0 && iy<255;++it){
  double end=std::min({1.,nx,ny}),td=std::abs(dx-dy)>1e-16?((y-iy)-(x-ix))/(dx-dy):2;
  double split=(td>enter+1e-12 && td<end-1e-12)?td:end,lo=enter;
  for(int k=0;k<2;++k){
   double hi=k==0?split:end;
   if(hi>lo+1e-14){
    double mid=.5*(lo+hi),fx=x-ix+dx*mid,fy=y-iy+dy*mid;
    double a=h[iy*256+ix],b=h[iy*256+ix+1],c0=h[(iy+1)*256+ix],d=h[(iy+1)*256+ix+1];
    double gx=fy>fx?d-c0:b-a,gy=fy>fx?c0-a:d-b;
    piece(s,lo,hi,a+gx*(x-ix)+gy*(y-iy)-z,gx*dx+gy*dy-dz);
   }
   lo=hi;if(lo>=end)break;
  }
  if(end>=1-1e-13){s.done=1;break;}
  bool ax=nx<=ny,ay=ny<=nx;if(ax){ix+=xs;nx+=sx;}if(ay){iy+=ys;ny+=sy;}enter=end;
 }
 return s;
}
extern "C" void c156_counts(const float*h,int tr,int tc,const int32_t* pix,int n,double*out){
 #pragma omp parallel for schedule(static)
 for(int j=0;j<n;++j){
  RayStat s=trace(h,tr,tc,pix[j]);
  out[j*5]=s.crossings;out[j*5+1]=s.blocked;out[j*5+2]=s.highest;out[j*5+3]=s.under;out[j*5+4]=s.done;
 }
}
