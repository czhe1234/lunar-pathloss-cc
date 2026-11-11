// Paired primary-ray diagnostic on the registered candidate mesh. No supervised targets here.
#include "c67_transport.cu"
__global__ void observe_offset(Scene s,const float3 *quad,int nq,float *result,int mode) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=faces*nq)return;
    float fx,fy;V point=face_point(s,id/nq,quad[id%nq],&fx,&fy);point.z+=1;
    V delta=point-s.tx;V dir=unit({delta.x*s.spacing,delta.y*s.spacing,delta.z});
    V gd={dir.x/s.spacing,dir.y/s.spacing,dir.z};
    // Area-stratified source directions. Their angular density includes every receiver-sheet preimage.
    V o=s.tx;o.z-=1;float rho=0,travel=0;
    for(int n=0;n<48;++n){
        Hit hit;if(!first_hit(s,o,dir,600-travel,false,hit))break;
        float t=travel+hit.distance,cosine=fmaxf(1e-7f,fabsf(dot3(dir,flat_normal(s,hit.face))));
        rho+=t*t/(s.geometry[hit.face].w*cosine);
        float step=hit.distance+.00035f;o=o+gd*step;travel+=step;
    }
    if(rho<1e-8f)return;
    float solid=1.f/(nq*rho);
    #pragma unroll
    for(int intervention=0;intervention<2;++intervention) {
        V o1=s.tx;
        for(int n=0;n<48;++n){
            Hit hit;if(!first_hit(s,o1,dir,600,true,hit)||!hit.receiver)break;
            V nrm=flat_normal(s,hit.face);float cosine=dot3(dir,nrm);
            float p=solid/(s.geometry[hit.face].w*fmaxf(1e-7f,fabsf(cosine)));
            atomicAdd(&result[hit.face*2+intervention],p);
            V end=o1+gd*hit.distance;
            if(!intervention){o1=end+gd*.00035f;continue;}
            float wx=end.x*s.spacing-128,wy=128-end.y*s.spacing;
            float mag=(1+fmaxf(fmaxf(fabsf(wx),fabsf(wy)),fabsf(end.z)))*8.94069671631e-5f;
            float sign=cosine>=0?1.f:-1.f;
            o1=end+V{nrm.x/s.spacing,nrm.y/s.spacing,nrm.z}*(sign*mag);
        }
    }
}
extern "C" int offset77(const float4 *geom,const float3 *quad,int nq,float x,float y,float z,float s,float top,float *out,cudaStream_t st) {
    Scene scene{geom,nullptr,nullptr,{x,y,z},s,top,0,0};
    observe_offset<<<(faces*nq+255)/256,256,0,st>>>(scene,quad,nq,out,0);return cudaGetLastError();
}
