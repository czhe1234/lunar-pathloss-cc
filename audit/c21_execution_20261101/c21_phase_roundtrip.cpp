#include <cmath>
#include <algorithm>
extern "C" void phase_map(const float* h,int tr,int tc,float* out){
 float ht=h[tr*256+tc],frac=ht-std::floor(ht),zt=ht+3.f;
 for(int r=0;r<256;r++)for(int c=0;c<256;c++){
  int idx=r*256+c;float dx=c-tc,dy=r-tr,dz=h[idx]+1.f-ht-3.f;
  float bins[8];for(int j=0;j<8;j++)bins[j]=-1e20f;
  float length=0,prev=-(3.f-frac),maximum=-1e20f,need=-1e20f;int positive=0,cross=0,best=0;
  for(int k=0;k<128;k++){
   float t=(k+.5f)/128.f;
   double x=tc+dx*t,y=tr+dy*t;
   x=((x/127.5-1.)+1.)*127.5;y=((y/127.5-1.)+1.)*127.5;
   x=std::max(0.,std::min(255.,x));y=std::max(0.,std::min(255.,y));
   int ix=std::min(254,(int)x),iy=std::min(254,(int)y);double u=x-ix,v=y-iy;
   float z=(float)((1-u)*(1-v)*h[iy*256+ix]+u*(1-v)*h[iy*256+ix+1]+(1-u)*v*h[(iy+1)*256+ix]+u*v*h[(iy+1)*256+ix+1]);
   float e=z-(zt+dz*t);float encoded=std::asinh(e/2.f)/3.f;e=2.f*std::sinh(encoded*3.f)+frac*(1.f-t);
   bins[k/16]=std::max(bins[k/16],e);positive+=e>0;
   if(e>maximum){maximum=e;best=k;}need=std::max(need,e/t);cross+=(prev>0)!=(e>0);
   float dt=k==0?1.f/256:1.f/128;
   length+=(std::max(prev,0.f)+std::max(e,0.f))/std::max(std::abs(prev)+std::abs(e),1e-9f)*dt;prev=e;
  }
  length+=std::max(prev,0.f)/std::max(std::abs(prev)+1.f,1e-9f)/256;cross+=prev>0;
  float dist=std::sqrt(dx*dx+dy*dy+(dz+frac)*(dz+frac));
  for(int k=0;k<8;k++)out[k*65536+idx]=std::asinh(bins[k]/2.f)/3.f;
  out[8*65536+idx]=std::log1p(length*dist)/6;out[9*65536+idx]=positive/128.f;
  out[10*65536+idx]=(best+.5f)/128;out[11*65536+idx]=std::asinh(need/2.f)/3;
  out[12*65536+idx]=cross;out[13*65536+idx]=3-frac;
 }
}
