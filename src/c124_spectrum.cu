#include "c67_transport.cu"
// Each receiver face owns its 256 bins. No target-dependent data enter this kernel.
__global__ void spectrum_receivers(Scene s,const float3 *dirs,int count,const float4 *sources,
                                  const int *ids,int n,float *bins) {
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
 int face=ids[j];float fx,fy;V p=face_point(s,face,make_float3(1.f/3,1.f/3,1.f/3),&fx,&fy);p.z+=1;
 float *out=bins+j*256;
 for(int b=0;b<256;++b)out[b]=0;
 V txdiff={(s.tx.x-p.x)*s.spacing,(s.tx.y-p.y)*s.spacing,s.tx.z-p.z};
 float direct=sqrtf(dot3(txdiff,txdiff));
 float azref=atan2f(txdiff.y,txdiff.x);
 for(int i=0;i<count;++i) {
  V dir=from3(dirs[i]);Hit hit;if(!first_hit(s,p,dir,1e5f,false,hit))continue;
  float4 source=sources[hit.face];
  V ng=flat_normal(s,hit.face),ns=shade_normal(s,hit.face,hit.fx,hit.fy);
  float cosine=-dot3(ns,dir)*source.z;
  if(cosine<=0)continue;
  float scale=4.f*cosine/(count*fmaxf(.0001f,fabsf(dot3(ng,dir))));
  V hitToTX={txdiff.x-dir.x*hit.distance,txdiff.y-dir.y*hit.distance,txdiff.z-dir.z*hit.distance};
  float extra=fmaxf(0.f,sqrtf(dot3(hitToTX,hitToTX))+hit.distance-direct);
  int delay=min(7,(int)floorf(log2f(1.f+extra)));
  float az=atan2f(dir.y,dir.x)-azref;
  const float twopi=6.283185307179586f;
  az=az-floorf(az/twopi)*twopi;
  int angle=min(15,(int)floorf(az*(16.f/twopi)));
  int k=delay*16+angle;
  out[k]+=source.x*scale;out[128+k]+=source.y*scale;
 }
}
extern "C" int path_spectrum(const float4 *geometry,const float3 *normals,const float *material,
const float3 *directions,int nd,const float4 *emission,const int *ids,int n,
float x,float y,float z,float spacing,float top,int anti,int smooth,float *bins,cudaStream_t stream) {
 Scene scene{geometry,normals,material,{x,y,z},spacing,top,anti,smooth};
 spectrum_receivers<<<(n+63)/64,64,0,stream>>>(scene,directions,nd,emission,ids,n,bins);
 return cudaGetLastError();
}
