#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <vector>
using Z=std::complex<double>;
using V=std::array<double,3>;
using E=std::array<Z,3>;
static double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
static V unit(V a){double v=std::sqrt(dot(a,a));return {a[0]/v,a[1]/v,a[2]/v};}
static Z edot(E a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
struct Hit{double t,sx,sy;};
static std::array<Z,2> transmission(double c,int band){
 const double f[2]={415e6,5.775e9},sig[2]={.0003562,.009324};
 c=std::max(1e-7,std::min(1.,c));
 Z eta(3.411,-sig[band]/(2*M_PI*f[band]*8.8541878128e-12));
 Z a=std::sqrt(eta-(1-c*c));
 Z rs=(c-a)/(c+a),rp=(eta*c-a)/(eta*c+a);
 Z e=std::exp(Z(0,-1)*(2*M_PI*f[band]/299792458.*10.)*a);
 return {(1.-rs*rs)*e/(1.-rs*rs*e*e),(1.-rp*rp)*e/(1.-rp*rp*e*e)};
}
extern "C" void triangle_transport_points(const double* h,int tr,int tc,const double* rx,int N,int diag,double* out){
 for(int j=0;j<N;j++){
  double rr=rx[j*3],cc=rx[j*3+1];double dx=cc-tc,dy=rr-tr;
  double zt=std::floor(h[tr*256+tc])+3.,zr=rx[j*3+2],dz=zr-zt;
  double dist=std::sqrt(dx*dx+dy*dy+dz*dz);V k={dx/dist,dy/dist,dz/dist};
  V ev={-k[2]*k[0],-k[2]*k[1],1.-k[2]*k[2]};
  if(dot(ev,ev)<1e-14)ev={1.,0.,0.};ev=unit(ev);V eh=unit(cross(k,ev));
  std::vector<double> tt{0.,1.};tt.reserve(512);
  if(dx!=0)for(int x=(int)std::floor(std::min((double)tc,cc))+1;x<std::max((double)tc,cc);x++)tt.push_back((x-tc)/dx);
  if(dy!=0)for(int y=(int)std::floor(std::min((double)tr,rr))+1;y<std::max((double)tr,rr);y++)tt.push_back((y-tr)/dy);
  std::sort(tt.begin(),tt.end());tt.erase(std::unique(tt.begin(),tt.end(),[](double a,double b){return std::abs(a-b)<1e-12;}),tt.end());
  {
   std::vector<Hit> hits;hits.reserve(32);
   for(size_t q=0;q+1<tt.size();q++){
    double lo=tt[q],hi=tt[q+1],mid=(lo+hi)/2;
    int ix=std::max(0,std::min(254,(int)std::floor(tc+dx*mid)));
    int iy=std::max(0,std::min(254,(int)std::floor(tr+dy*mid)));
    double a=h[iy*256+ix],b=h[iy*256+ix+1],c=h[(iy+1)*256+ix],d=h[(iy+1)*256+ix+1];
    for(int side=0;side<2;side++){
     double sx,sy,anchor;
     if(diag==0){
      if(side==0){sx=b-a;sy=d-b;anchor=a;}
      else{sx=d-c;sy=c-a;anchor=a;}
     }else{
      if(side==0){sx=b-a;sy=c-a;anchor=a;}
      else{sx=d-c;sy=d-b;anchor=b+c-d;}
     }
     double den=dz-sx*dx-sy*dy;
     if(std::abs(den)<1e-13)continue;
     double t=(anchor+sx*(tc-ix)+sy*(tr-iy)-zt)/den;
     if(t<=lo+1e-11 || t>hi+1e-11 || t<=0 || t>=1)continue;
     double u=tc+dx*t-ix,v=tr+dy*t-iy;
     bool valid=diag==0?(side==0?v<=u+1e-9:v>=u-1e-9):(side==0?u+v<=1+1e-9:u+v>=1-1e-9);
     if(valid)hits.push_back({t,sx,sy});
    }
   }
   std::sort(hits.begin(),hits.end(),[](Hit a,Hit b){return a.t<b.t;});
   hits.erase(std::unique(hits.begin(),hits.end(),[](Hit a,Hit b){return std::abs(a.t-b.t)<1e-9;}),hits.end());
   double* o=out+j*16;std::fill(o,o+16,0.);
   o[0]=hits.size();o[1]=1.;o[4]=1.;o[5]=dist;o[14]=1.;o[15]=0.;
   E fields[2][2];double loss[2][2]={{0,0},{0,0}},reset[2]={0,0},te[2]={0,0},tm[2]={0,0};
   for(int b=0;b<2;b++)for(int v=0;v<3;v++){fields[b][0][v]=ev[v];fields[b][1][v]=eh[v];}
   V prior={0,0,0};
   for(size_t q=0;q<hits.size();q++){
    Hit hit=hits[q];V normal=unit({-hit.sx,-hit.sy,1.});double cosine=std::abs(dot(k,normal));
    o[1]=std::min(o[1],cosine);o[2]+=-std::log(std::max(cosine,1e-7));o[4]=std::min(o[4],std::abs(cosine-1/std::sqrt(1+3.411)));
    if(q)o[5]=std::min(o[5],(hit.t-hits[q-1].t)*dist);
    if(q==0)o[14]=hit.t;o[15]=hit.t;
    V s=cross(k,normal);if(dot(s,s)<1e-14)s=eh;else s=unit(s);V pp=unit(cross(s,k));
    if(q)o[3]+=1-std::abs(dot(s,prior));prior=s;
    for(int b=0;b<2;b++){
     auto t=transmission(cosine,b);double ts2=std::norm(t[0]),tp2=std::norm(t[1]);
     reset[b]+=-10*std::log10(std::max((ts2+tp2)/2,1e-300));
     te[b]+=-10*std::log10(std::max(ts2,1e-300));tm[b]+=-10*std::log10(std::max(tp2,1e-300));
     for(int pol=0;pol<2;pol++){
      Z es=edot(fields[b][pol],s),ep=edot(fields[b][pol],pp);E v;double power=0;
      for(int k=0;k<3;k++){v[k]=t[0]*es*s[k]+t[1]*ep*pp[k];power+=std::norm(v[k]);}
      power=std::max(power,1e-300);loss[b][pol]+=-10*std::log10(power);
      for(int k=0;k<3;k++)fields[b][pol][k]=v[k]/std::sqrt(power);
     }
    }
   }
   for(int b=0;b<2;b++){
    int c=6+b*4;o[c]=loss[b][0];o[c+1]=loss[b][1];o[c+2]=reset[b];
    double mn=std::min(te[b],tm[b]);o[c+3]=mn-10*std::log10((std::pow(10.,-(te[b]-mn)/10)+std::pow(10.,-(tm[b]-mn)/10))/2);
   }
  }
 }
}
