// C135 numerical refinement of existing C92 MIS: area quadrature at each RX facet.
#include "c92_diffuse.cu"
__global__ void receiver135(Scene s,const int*ids,int count,const float3*bary,int nb,
 const float3*dirs,int nd,int nl,const float4*field,const float*prob,const float*cdf,
 unsigned seed,float4*out){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count)return;
 int f=ids[j];float4 sum=make_float4(0,0,0,0);const float pi=3.14159265358979323846f;
 for(int qidx=0;qidx<nb;qidx++){
  float fx,fy;V p=face_point(s,f,bary[qidx],&fx,&fy);p.z+=1;
  V own=shade_normal(s,f,fx,fy);float4 res=make_float4(0,0,0,0);
  unsigned seedq=seed+qidx*9973U;
  float rot=2*pi*u78(seedq^mix78(f));float cs=cosf(rot),sn=sinf(rot);
  for(int i=0;i<nd;i++){
   V u=from3(dirs[i]);V dir={u.x*cs-u.y*sn,u.y*cs+u.x*sn,u.z};
   Hit h;if(!first_hit(s,p,dir,500.f,false,h))continue;
   V ng=flat_normal(s,h.face);float cosine=fmaxf(1e-5f,fabsf(dot3(ng,dir)));
   float r2=fmaxf(.0001f,h.distance*h.distance);
   float qr=cosine/(4*pi*r2),ql=prob[h.face]/s.geometry[h.face].w;
   float bal=nd*qr/fmaxf(1e-28f,nd*qr+nl*ql);
   gather92(s,false,own,dir,field,h.face,h.fx,h.fy,4*bal/(nd*cosine),res);
  }
  for(int i=0;i<nl;i++){
   unsigned r=seedq^mix78(f*7919U+i*8191U);int target=sample78(cdf,(i+u78(r))/nl);
   float t=sqrtf(u78(r+1)),a=u78(r+2),x,y;
   V v=face_point(s,target,make_float3(1-t,t*(1-a),t*a),&x,&y);
   V d=v-p;V phys={d.x*s.spacing,d.y*s.spacing,d.z};float r2=fmaxf(.0001f,dot3(phys,phys));
   float length=sqrtf(r2);V dir=phys*(1/length);Hit h;
   if(first_hit(s,p,dir,length-.001f,false,h))continue;
   V ng=flat_normal(s,target);float ql=prob[target]/s.geometry[target].w,qr=fabsf(dot3(ng,dir))/(4*pi*r2);
   float bal=nl*ql/fmaxf(1e-28f,nl*ql+nd*qr);
   gather92(s,false,own,dir,field,target,x,y,bal/(pi*r2*nl*fmaxf(1e-28f,ql)),res);
  }
  sum.x+=res.x/nb;sum.y+=res.y/nb;
 }
 out[j]=sum;
}
extern "C" int c135_mis(const float4*g,const float3*n,const float*mat,const int*ids,int count,
 const float3*bary,int nb,const float3*dirs,int nd,int nl,const float4*field,
 const float*prob,const float*cdf,int seed,float top,float4*out,cudaStream_t st){
 Scene s{g,n,mat,{0,0,0},256.f/255,top,0,1};
 receiver135<<<(count+63)/64,64,0,st>>>(s,ids,count,bary,nb,dirs,nd,nl,field,prob,cdf,(unsigned)seed,out);
 return cudaGetLastError();
}
