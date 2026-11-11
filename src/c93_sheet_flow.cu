// Deterministic launch integral over *first* sheet hits. Multiple intersections are outputs, not additional sources.
#include "c67_transport.cu"
__global__ void flow93(Scene s,const float3*quad,int nq,float *flux){
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=faces*nq)return;
    int f=id/nq;float fx,fy;V p=face_point(s,f,quad[id%nq],&fx,&fy);p.z+=1.f;
    V delta=p-s.tx;V dr={delta.x*s.spacing,delta.y*s.spacing,delta.z};
    float d=sqrtf(fmaxf(1e-10f,dot3(dr,dr)));V dir=dr*(1/d);
    Hit h;if(first_hit(s,s.tx,dir,d-.001f,true,h))return;
    V gn=flat_normal(s,f);float cos0=fabsf(dot3(gn,dir));
    float base=1/(nq*fmaxf(.0625f,d*d));float omega=s.geometry[f].w*cos0*base;
    atomicAdd(flux+6*f,base);atomicAdd(flux+6*f+5,1.f/nq);
    V gdir={dir.x/s.spacing,dir.y/s.spacing,dir.z};
    for(int mode=0;mode<2;++mode){
        V pos=p,n=gn;int previous=f;
        for(int j=0;j<32;++j){
            if(mode){
                const float world_scale=1+fmaxf(fabsf(pos.x*s.spacing-128),fmaxf(fabsf(128-pos.y*s.spacing),fabsf(pos.z)));
                const float step=copysignf(world_scale*(1500.f/16777216.f),dot3(n,dir));
                pos=pos+V{n.x/s.spacing,n.y/s.spacing,n.z}*step;
            } else pos=pos+gdir*.00025f;
            if(!first_hit(s,pos,dir,450.f,true,h)||!h.receiver)break;
            if(h.face==previous&&h.distance<.008f)break;
            n=flat_normal(s,h.face);float co=fmaxf(1e-6f,fabsf(dot3(n,dir)));
            float power=omega/(s.geometry[h.face].w*co);
            if(isfinite(power)){atomicAdd(flux+6*h.face+1+mode,power);
                if(mode){atomicAdd(flux+6*h.face+3,omega);atomicAdd(flux+6*h.face+4,1.f/nq);}}
            pos=pos+gdir*h.distance;previous=h.face;
        }
    }
}
extern "C" int receiver_flow93(const float4*g,const float3*q,int nq,float tx,float ty,float tz,float top,float*out,cudaStream_t st){
    Scene s{g,nullptr,nullptr,{tx,ty,tz},256.f/255.f,top,0,0};
    flow93<<<(faces*nq+127)/128,128,0,st>>>(s,q,nq,out);return cudaGetLastError();
}
