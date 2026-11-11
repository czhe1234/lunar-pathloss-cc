// CC C131: expose the unavoidable direct component on the fixed candidate mesh.
// Reuses CC's independently implemented native intersection and polarization transport.
#include "c67_transport.cu"
__global__ void direct131(Scene s,const float3 *bary,int nq,float4 *output){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=faces)return;
 float l=0,h=0,los=0,visible=0;
 for(int j=0;j<nq;++j){
  float x,y;V p=face_point(s,f,bary[j],&x,&y);p.z+=1;
  Beam b=incident(s,p);float inv=1/fmaxf(.0625f,b.length*b.length);
  l+=(b.pol[0].h.power()+b.pol[0].v.power())*inv;
  h+=(b.pol[1].h.power()+b.pol[1].v.power())*inv;
  if(b.crossings==0){los+=inv;visible+=1;}
 }
 output[f]=make_float4(l/nq,h/nq,los/nq,visible/nq);
}
extern "C" int c131_direct(const float4*g,const float3*n,const float*mat,const float3*q,int nq,
 float x,float y,float z,float spacing,float top,float4*out,cudaStream_t stream){
 Scene s{g,n,mat,{x,y,z},spacing,top,0,1};
 direct131<<<(faces+127)/128,128,0,stream>>>(s,q,nq,out);
 return cudaGetLastError();
}
