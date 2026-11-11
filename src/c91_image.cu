// Enumerate specular paths by reflecting TX in every candidate terrain triangle.
#include "c67_transport.cu"
constexpr int K91=14;
__device__ V physical91(V v,float s){return {v.x*s,v.y*s,v.z};}
__global__ void mirrors91(Scene s,float4 *images,int tangent){
    int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=faces)return;
    float fx,fy;V p=face_point(s,f,{1.f/3,1.f/3,1.f/3},&fx,&fy);
    V n=tangent?shade_normal(s,f,fx,fy):flat_normal(s,f);
    float d=dot3(physical91(s.tx-p,s.spacing),n);
    images[f]=make_float4(s.tx.x-2*d*n.x/s.spacing,s.tx.y-2*d*n.y/s.spacing,s.tx.z-2*d*n.z,d);
}
__global__ void reflect91(Scene s,const float4 *images,const float3 *targets,int n,int tangent,float *out){
    int p=blockIdx.y;if(p>=n)return;V r=from3(targets[p]);float *dst=out+p*K91;
    float d02=dot3(physical91(r-s.tx,s.spacing),physical91(r-s.tx,s.spacing));
    for(int f=blockIdx.x*blockDim.x+threadIdx.x;f<faces;f+=blockDim.x*gridDim.x){
        int cell=f%cells,ix=cell%255,iy=cell/255;float4 geom=s.geometry[f],im=images[f];
        // Intersection of the image-to-RX line with the actual terrain plane.
        float dx=r.x-im.x,dy=r.y-im.y,dz=r.z-im.z;
        float den=dz-geom.y*dx-geom.z*dy;
        float num=geom.x+geom.y*(im.x-ix)+geom.z*(im.y-iy)-im.z;
        if(fabsf(den)<1e-10f)continue;
        float t=num/den;if(!(t>0&&t<1))continue;
        float fx=fmaf(t,dx,im.x)-ix,fy=fmaf(t,dy,im.y)-iy;
        if(fx<0||fx>=1||fy<0||fy>=1||((fy>fx)!=(f>=cells)))continue;
        V q={ix+fx,iy+fy,fmaf(geom.y,fx,fmaf(geom.z,fy,geom.x))};
        V plane=flat_normal(s,f);
        float ts=dot3(physical91(s.tx-q,s.spacing),plane),rs=dot3(physical91(r-q,s.spacing),plane);
        if(ts*rs<=0)continue;
        V outgoing=physical91(r-q,s.spacing);float len=sqrtf(dot3(outgoing,outgoing));if(len<=.02f)continue;
        outgoing=outgoing*(1/len);
        V incoming=physical91(q-s.tx,s.spacing);float len1=sqrtf(dot3(incoming,incoming));if(len1<.02f)continue;
        atomicAdd(dst+0,1.f);
        Hit hit;if(first_hit(s,q,outgoing,len-.001f,false,hit))continue;
        atomicAdd(dst+1,1.f);
        auto beam=incident(s,q);V normal=tangent?shade_normal(s,f,fx,fy):plane;
        float scale=(1.f-.632f*.632f)*d02/fmaxf(.0625f,(len+len1)*(len+len1));
        float lo=scale*reflected(s,beam,normal,0),hi=scale*reflected(s,beam,normal,1);
        if(beam.crossings==0){
            atomicAdd(dst+2,1.f);atomicAdd(dst+3,lo);atomicAdd(dst+4,hi);
        }
        atomicAdd(dst+5,lo);atomicAdd(dst+6,hi);
        atomicMax((int*)(dst+7),__float_as_int(fmaxf(lo,0.f)));atomicMax((int*)(dst+8),__float_as_int(fmaxf(hi,0.f)));
        float excess=len+len1-sqrtf(fmaxf(d02,0));
        atomicAdd(dst+9,hi*log1pf(fmaxf(0.f,excess)));atomicAdd(dst+10,hi*log1pf(len));
        atomicAdd(dst+11,lo*log1pf(fmaxf(0.f,excess)));atomicAdd(dst+12,lo*log1pf(len));
        atomicAdd(dst+13,hi*hi);
    }
}
extern "C" int image91(const float4 *g,const float3 *vn,const float *mat,const float3 *targets,int n,
  float x,float y,float z,float top,int tangent,float4 *images,float *out,cudaStream_t st){
    Scene s{g,vn,mat,{x,y,z},256.f/255.f,top,0,1};
    mirrors91<<<(faces+127)/128,128,0,st>>>(s,images,tangent);
    reflect91<<<dim3(32,n),128,0,st>>>(s,images,targets,n,tangent,out);return cudaGetLastError();
}
