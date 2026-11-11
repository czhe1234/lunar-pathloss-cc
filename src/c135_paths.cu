// C135: continuous specular chains, polarized diffuse sources and reverse specular legs.
// Uses the frozen candidate mesh. No targets/masks are accepted by these kernels.
#include "c67_transport.cu"
__device__ C conjc(C a){return {a.re,-a.im};}
__device__ C addc(C a,C b){return a+b;}
struct Mat {C a,b,c,d;};
__device__ Mat eye(){return {{1,0},{0,0},{0,0},{1,0}};}
__device__ Mat mm(Mat A,Mat B){return {
 A.a*B.a+A.b*B.c,A.a*B.b+A.b*B.d,
 A.c*B.a+A.d*B.c,A.c*B.b+A.d*B.d};}
__device__ void basis(V k,V&h,V&v){
 h=unit({-k.y,k.x,0});if(dot3(h,h)<.5f)h={0,1,0};v=cross3(h,k);
}
__device__ void refl_coeff(const float *lut,float cosine,int band,C&rs,C&rp){
 float u=fminf(4095.9999f,fmaxf(0.f,cosine*4096));int j=min(4095,int(u));float a=u-j;
 const float*r=lut+(band*4097+j)*4;float t[4];
 for(int k=0;k<4;k++)t[k]=fmaf(a,r[k+4]-r[k],r[k]);
 rs={t[0],t[1]};rp={t[2],t[3]};
}
__device__ Mat reflection_matrix(const Scene&s,V ki,V ko,V n,const float*lut,int band){
 V hi,vi,ho,vo;basis(ki,hi,vi);basis(ko,ho,vo);
 V te=cross3(ki,n);
 te=dot3(te,te)>1e-12f?unit(te):hi;
 V tmi=cross3(te,ki),tmo=cross3(te,ko);
 float a=dot3(te,hi),b=dot3(te,vi),c=dot3(tmi,hi),d=dot3(tmi,vi);
 float u=dot3(ho,te),v=dot3(ho,tmo),w=dot3(vo,te),x=dot3(vo,tmo);
 C rs,rp;refl_coeff(lut,fabsf(dot3(ki,n)),band,rs,rp);
 // Preserve the old material intensity interpolation while retaining complex phase.
 C ts,tp;float ri,pi;coefficients(s,fabsf(dot3(ki,n)),band,ts,tp,ri,pi);
 rs=rs*sqrtf(ri/fmaxf(1e-35f,rs.power()));rp=rp*sqrtf(pi/fmaxf(1e-35f,rp.power()));
 float scat=sqrtf(1-.632f*.632f);rs=rs*scat;rp=rp*scat;
 return {rs*(u*a)+rp*(v*c),rs*(u*b)+rp*(v*d),
         rs*(w*a)+rp*(x*c),rs*(w*b)+rp*(x*d)};
}
__device__ Beam reflect_beam(const Scene&s,Beam b,V n,const float*lut){
 V ko=b.dir-n*(2*dot3(b.dir,n));
 for(int f=0;f<2;f++){
  Mat M=reflection_matrix(s,b.dir,ko,n,lut,f);
  Pol p=b.pol[f];b.pol[f]={M.a*p.h+M.b*p.v,M.c*p.h+M.d*p.v};
 }
 b.dir=ko;basis(ko,b.h,b.v);return b;
}
__device__ int emidx(int prefix,int face,int side,int band){
 return (((prefix*faces+face)*2+side)*2+band)*4;
}
__device__ void scatter_deposit(const Scene&s,Beam b,V n,float density,
 int prefix,int face,float*em,bool atom){
 int side=-dot3(b.dir,n)>=0?0:1;
 for(int f=0;f<2;f++){
  Pol p=b.pol[f];float total=p.h.power()+p.v.power();
  float scale=.632f*.632f*density*reflected(s,b,n,f)/fmaxf(total,1e-35f);
  // Official XPD=0.1 is a deterministic rotation in (theta,phi), not depolarization.
  C h=p.h*sqrtf(.9f)+p.v*sqrtf(.1f);
  C v=p.h*(-sqrtf(.1f))+p.v*sqrtf(.9f);C hv=h*conjc(v);
  float val[4]={h.power()*scale,v.power()*scale,hv.re*scale,hv.im*scale};
  int idx=emidx(prefix,face,side,f);
  for(int j=0;j<4;j++){if(atom)atomicAdd(em+idx+j,val[j]);else em[idx+j]=val[j];}
 }
}
__global__ void source_zero(Scene s,float*em){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=faces)return;
 float x,y;V p=face_point(s,f,make_float3(1.f/3,1.f/3,1.f/3),&x,&y);
 Beam b=incident(s,p);V n=shade_normal(s,f,x,y);
 float density=fabsf(dot3(b.dir,flat_normal(s,f)))/fmaxf(.0625f,b.length*b.length);
 scatter_deposit(s,b,n,density,0,f,em,false);
}
__global__ void spec_chains(Scene s,const float*lut,const float3*bary,int ns,
 int depth,int maxsteps,float*received,float*em){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=faces*ns)return;
 int f=id/ns,j=id%ns;float x,y;V p=face_point(s,f,bary[j],&x,&y);
 Beam b=incident(s,p);float packet=s.geometry[f].w*fabsf(dot3(b.dir,flat_normal(s,f)))/
      (ns*fmaxf(.0625f,b.length*b.length));
 b=reflect_beam(s,b,shade_normal(s,f,x,y),lut);
 int order=1;
 for(int step=0;step<maxsteps;step++){
  if(order>depth)break;
  float e0=b.pol[0].h.power()+b.pol[0].v.power(),e1=b.pol[1].h.power()+b.pol[1].v.power();
  if(fmaxf(e0,e1)*packet<1e-30f)break;
  Hit hit;if(!first_hit(s,p,b.dir,1e5f,true,hit))break;
  V gd={b.dir.x/s.spacing,b.dir.y/s.spacing,b.dir.z};
  V where=p+gd*hit.distance;
  if(hit.receiver){
   float den=s.geometry[hit.face].w*fmaxf(1e-7f,fabsf(dot3(b.dir,flat_normal(s,hit.face))));
   int idx=(hit.face*depth+order-1)*2;
   atomicAdd(received+idx,e0*packet/den);atomicAdd(received+idx+1,e1*packet/den);
   p=p+gd*(hit.distance+.00035f);
  }else{
   V n=shade_normal(s,hit.face,hit.fx,hit.fy);
   scatter_deposit(s,b,n,packet/s.geometry[hit.face].w,order,hit.face,em,true);
   if(order==depth)break;
   b=reflect_beam(s,b,n,lut);order++;p=where;
  }
 }
}
__device__ float measure(Mat A,const float*e){
 float h=A.a.power()+A.c.power(),v=A.b.power()+A.d.power();
 C cross=A.a*conjc(A.b)+A.c*conjc(A.d);
 return fmaxf(0.f,h*e[0]+v*e[1]+2*(cross.re*e[2]-cross.im*e[3]));
}
__global__ void mixed_query(Scene s,const float*lut,const int*ids,const float*weights,
 int nq,const float3*bary,int nb,const float3*dirs,int nd,int pre,int post,
 const float*em,float*out){
 int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=nq)return;
 // Components: Q0-A0, Q(spec)-A0, Q0-A(spec), Q(spec)-A(spec), with two bands.
 float result[8]={0,0,0,0,0,0,0,0};
 for(int donor=0;donor<3;donor++){
  int face=ids[q*3+donor];float iw=weights[q*3+donor];if(iw<=0)continue;
  for(int j=0;j<nb;j++){
   float x,y;V origin=face_point(s,face,bary[j],&x,&y);origin.z+=1;
   for(int k=0;k<nd;k++){
    V dir=from3(dirs[k]),p=origin;Mat accum[2]={eye(),eye()};
    for(int bounce=0;bounce<=post;bounce++){
     Hit hit;if(!first_hit(s,p,dir,1e5f,false,hit))break;
     V n=shade_normal(s,hit.face,hit.fx,hit.fy),ng=flat_normal(s,hit.face);
     int side=-dot3(dir,n)>=0?0:1;
     float scale=4.f*iw*fabsf(dot3(dir,n))/(nb*nd*fmaxf(.0001f,fabsf(dot3(dir,ng))));
     for(int prefix=0;prefix<=pre;prefix++){
      int part=(bounce>0?2:0)+(prefix>0?1:0);
      for(int band=0;band<2;band++)
       result[part*2+band]+=scale*measure(accum[band],em+emidx(prefix,hit.face,side,band));
     }
     if(bounce==post)break;
     V next=dir-n*(2*dot3(dir,n));
     for(int band=0;band<2;band++){
      Mat reflection=reflection_matrix(s,next*(-1),dir*(-1),n,lut,band);
      accum[band]=mm(accum[band],reflection);
     }
     V gd={dir.x/s.spacing,dir.y/s.spacing,dir.z};
     p=p+gd*hit.distance;dir=next;
    }
   }
  }
 }
 for(int j=0;j<8;j++)out[q*8+j]=result[j];
}
extern "C" int c135_forward(const float4*g,const float3*n,const float*material,
 const float*refl,const float3*bary,int ns,int depth,int maxsteps,float x,float y,float z,float top,
 float*spec,float*em,cudaStream_t stream){
 Scene s{g,n,material,{x,y,z},256.f/255,top,0,1};
 source_zero<<<(faces+127)/128,128,0,stream>>>(s,em);
 spec_chains<<<(faces*ns+127)/128,128,0,stream>>>(s,refl,bary,ns,depth,maxsteps,spec,em);
 return cudaGetLastError();
}
extern "C" int c135_query(const float4*g,const float3*n,const float*material,
 const float*refl,const int*ids,const float*weights,int nq,const float3*bary,int nb,
 const float3*dirs,int nd,int pre,int post,float top,const float*em,float*out,cudaStream_t stream){
 Scene s{g,n,material,{0,0,0},256.f/255,top,0,1};
 mixed_query<<<(nq+63)/64,64,0,stream>>>(s,refl,ids,weights,nq,bary,nb,dirs,nd,pre,post,em,out);
 return cudaGetLastError();
}

extern "C" int c135_direct(const float4*g,const float3*n,const float*material,
 const float3*bary,int ns,float x,float y,float z,float top,float4*out,float4*tmp,cudaStream_t stream){
 Scene s{g,n,material,{x,y,z},256.f/255,top,0,1};
 primary_faces<<<(faces+127)/128,128,0,stream>>>(s,bary,ns,out,tmp);return cudaGetLastError();
}

__global__ void refl_audit(Scene s,const float*lut,const float3*bary,int ns,float*out){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=faces*ns)return;
 float x,y;int face=id/ns;V p=face_point(s,face,bary[id%ns],&x,&y);Beam b=incident(s,p);
 V n=shade_normal(s,face,x,y);Beam next=reflect_beam(s,b,n,lut);
 float weight=s.geometry[face].w*fabsf(dot3(b.dir,flat_normal(s,face)))/(ns*fmaxf(.0625f,b.length*b.length));
 for(int f=0;f<2;f++){
  out[id*4+f]=weight*(1-.632f*.632f)*reflected(s,b,n,f);
  out[id*4+f+2]=weight*(next.pol[f].h.power()+next.pol[f].v.power());
 }
}
extern "C" int c135_refl_audit(const float4*g,const float3*n,const float*material,const float*refl,const float3*bary,int ns,
 float x,float y,float z,float top,float*out,cudaStream_t stream){
 Scene s{g,n,material,{x,y,z},256.f/255,top,0,1};
 refl_audit<<<(faces*ns+127)/128,128,0,stream>>>(s,refl,bary,ns,out);return cudaGetLastError();
}

extern "C" int c135_reference_spec(const float4*g,const float3*n,const float*material,
 const float3*bary,int ns,float x,float y,float z,float top,float*out,cudaStream_t stream){
 Scene s{g,n,material,{x,y,z},256.f/255,top,0,1};
 specular_beams<<<(faces*ns+127)/128,128,0,stream>>>(s,bary,ns,out);return cudaGetLastError();
}
