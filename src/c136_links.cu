// Native-facet source-state experiment. Sparse transport is input-only and fixed.
#include "c135_paths.cu"
__device__ unsigned hash136(unsigned x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;return x;}
__device__ float u136(unsigned x){return (hash136(x)>>8)*(1.f/16777216);}
__device__ int sample136(const float*cdf,float v){int lo=0,hi=faces-1;while(lo<hi){int m=(lo+hi)/2;if(cdf[m]<v)lo=m+1;else hi=m;}return lo;}
__device__ void record136(int row,int src,float lo,float hi,int cap,int*count,int2*index,float2*value){
 if(fmaxf(lo,hi)<=1e-30f)return;
 int j=atomicAdd(count,1);if(j<cap){index[j]=make_int2(row,src);value[j]=make_float2(lo,hi);}
}
__global__ void spec136(Scene s,const float*lut,const float3*bary,int ns,int depth,
 const int*lookup,int cap,int*count,int2*ix,float2*value){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=faces*ns)return;
 int first=id/ns;float x,y;V p=face_point(s,first,bary[id%ns],&x,&y);
 Beam b=incident(s,p);float packet=s.geometry[first].w*fabsf(dot3(b.dir,flat_normal(s,first)))/(ns*fmaxf(.0625f,b.length*b.length));
 b=reflect_beam(s,b,shade_normal(s,first,x,y),lut);int order=1;
 for(int step=0;step<128;step++){
  if(order>depth)break;
  float e0=b.pol[0].h.power()+b.pol[0].v.power(),e1=b.pol[1].h.power()+b.pol[1].v.power();
  if(fmaxf(e0,e1)*packet<1e-30f)break;
  Hit hit;if(!first_hit(s,p,b.dir,1e5f,true,hit))break;
  V gd={b.dir.x/s.spacing,b.dir.y/s.spacing,b.dir.z};V where=p+gd*hit.distance;
  if(hit.receiver){
   int row=lookup[hit.face];
   if(row>=0){float den=s.geometry[hit.face].w*fmaxf(1e-7f,fabsf(dot3(b.dir,flat_normal(s,hit.face))));
    record136(row,first,e0*packet/den,e1*packet/den,cap,count,ix,value);}
   p=p+gd*(hit.distance+.00035f);
  }else{if(order==depth)break;b=reflect_beam(s,b,shade_normal(s,hit.face,hit.fx,hit.fy),lut);order++;p=where;}
 }
}
__global__ void a136(Scene s,const float*lut,const int*ids,int nq,const float3*bary,int nb,
 const float3*dirs,int nd,int nl,int post,const float*coh,const float*prob,const float*cdf,
 unsigned seed,int cap,int*count,int2*ix,float2*value){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=nq)return;
 int face=ids[row];const float pi=3.14159265358979323846f;
 for(int bq=0;bq<nb;bq++){
  float x,y;V origin=face_point(s,face,bary[bq],&x,&y);origin.z+=1;
  unsigned seedq=seed+bq*9973U;float rot=2*pi*u136(seedq^hash136(face)),cs=cosf(rot),sn=sinf(rot);
  for(int k=0;k<nd;k++){
   V u=from3(dirs[k]);V dir={u.x*cs-u.y*sn,u.y*cs+u.x*sn,u.z},p=origin;
   Mat accum[2]={eye(),eye()};
   for(int bounce=0;bounce<=post;bounce++){
    Hit h;if(!first_hit(s,p,dir,500.f,false,h))break;
    V ng=flat_normal(s,h.face),n=shade_normal(s,h.face,h.fx,h.fy);
    int side=-dot3(dir,n)>=0?0:1;
    float cosine=fmaxf(1e-5f,fabsf(dot3(ng,dir))),r2=fmaxf(.0001f,h.distance*h.distance),balance=1;
    if(bounce==0){float qr=cosine/(4*pi*r2),ql=prob[h.face]/s.geometry[h.face].w;balance=nd*qr/fmaxf(1e-28f,nd*qr+nl*ql);}
    float scale=4*balance*fabsf(dot3(dir,n))/(nb*nd*cosine);
    int idx=(h.face*2+side)*8;
    record136(row,h.face*2+side,scale*measure(accum[0],coh+idx),scale*measure(accum[1],coh+idx+4),cap,count,ix,value);
    if(bounce==post)break;
    V next=dir-n*(2*dot3(dir,n));
    for(int f=0;f<2;f++)accum[f]=mm(accum[f],reflection_matrix(s,next*(-1),dir*(-1),n,lut,f));
    V gd={dir.x/s.spacing,dir.y/s.spacing,dir.z};p=p+gd*h.distance;dir=next;
   }
  }
  for(int k=0;k<nl;k++){
   unsigned r=seedq^hash136(face*7919U+k*8191U);int source=sample136(cdf,(k+u136(r))/nl);
   float t=sqrtf(u136(r+1)),a=u136(r+2),fx,fy;
   V q=face_point(s,source,make_float3(1-t,t*(1-a),t*a),&fx,&fy);
   V d=q-origin;V physical={d.x*s.spacing,d.y*s.spacing,d.z};float r2=fmaxf(.0001f,dot3(physical,physical)),length=sqrtf(r2);V dir=physical*(1/length);
   Hit h;if(first_hit(s,origin,dir,length-.001f,false,h))continue;
   V ng=flat_normal(s,source),n=shade_normal(s,source,fx,fy);int side=-dot3(dir,n)>=0?0:1;
   float ql=prob[source]/s.geometry[source].w,qr=fabsf(dot3(ng,dir))/(4*pi*r2);
   float bal=nl*ql/fmaxf(1e-28f,nl*ql+nd*qr);
   float scale=bal*fabsf(dot3(n,dir))/(nb*pi*r2*nl*fmaxf(1e-28f,ql));
   int idx=(source*2+side)*8;
   record136(row,source*2+side,scale*(coh[idx]+coh[idx+1]),scale*(coh[idx+4]+coh[idx+5]),cap,count,ix,value);
  }
 }
}
extern "C" int c136_spec(const float4*g,const float3*n,const float*mat,const float*refl,
 const float3*bary,int ns,int depth,float x,float y,float z,float top,const int*lookup,
 int cap,int*count,int2*ix,float2*value,cudaStream_t stream){
 Scene s{g,n,mat,{x,y,z},256.f/255,top,0,1};
 spec136<<<(faces*ns+127)/128,128,0,stream>>>(s,refl,bary,ns,depth,lookup,cap,count,ix,value);return cudaGetLastError();
}
extern "C" int c136_a(const float4*g,const float3*n,const float*mat,const float*refl,
 const int*ids,int nq,const float3*bary,int nb,const float3*dirs,int nd,int nl,int post,
 const float*coh,const float*prob,const float*cdf,int seed,float top,int cap,int*count,int2*ix,float2*value,cudaStream_t stream){
 Scene s{g,n,mat,{0,0,0},256.f/255,top,0,1};
 a136<<<(nq+63)/64,64,0,stream>>>(s,refl,ids,nq,bary,nb,dirs,nd,nl,post,coh,prob,cdf,(unsigned)seed,cap,count,ix,value);return cudaGetLastError();
}
