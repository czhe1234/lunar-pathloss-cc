// One-bounce diagnostics on selected receiver faces. The two sampling proposals integrate the same Lambertian term.
#include "c67_transport.cu"
__device__ unsigned mix78(unsigned v) {
    v^=v>>16;v*=0x7feb352dU;v^=v>>15;v*=0x846ca68bU;v^=v>>16;return v;
}
__device__ float u78(unsigned v){return (float(mix78(v)>>8)+.5f)*(1.f/16777216.f);}
__device__ int sample78(const float *cdf,float u){
    int a=0,b=faces-1;while(a<b){int m=(a+b)/2;if(cdf[m]<u)a=m+1;else b=m;}return a;
}
__global__ void new_diffuse78(Scene s,const int *ids,int count,const float3 *dirs,int nd,int nl,const float4 *em,
 const float *prob,const float *cdf,unsigned seed,float *res){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count)return;int f=ids[j];
    float x,y;V p=face_point(s,f,make_float3(1.f/3,1.f/3,1.f/3),&x,&y);p.z+=1;
    float reverseL=0,reverseH=0,misL=0,misH=0,lightL=0,lightH=0;
    const float pi=3.14159265358979323846f;
    for(int i=0;i<nd;++i){
        V dir=from3(dirs[i]);Hit h;if(!first_hit(s,p,dir,1e5f,false,h))continue;
        V ng=flat_normal(s,h.face),ns=shade_normal(s,h.face,h.fx,h.fy);float4 source=em[h.face];
        float cosine=-dot3(ns,dir)*source.z;if(cosine<=0)continue;
        float ngcos=fmaxf(1e-5f,fabsf(dot3(ng,dir)));
        float scale=4*cosine/(nd*ngcos);
        float qr=ngcos/(4*pi*fmaxf(.0001f,h.distance*h.distance)),ql=prob[h.face]/s.geometry[h.face].w;
        float balance=nd*qr/fmaxf(1e-30f,nd*qr+nl*ql);
        reverseL+=source.x*scale;reverseH+=source.y*scale;
        misL+=source.x*scale*balance;misH+=source.y*scale*balance;
    }
    for(int i=0;i<nl;++i){
        unsigned base=seed^mix78(f*7919U+i*8191U);float stratum=(i+u78(base))/nl;
        int target=sample78(cdf,stratum);float t=sqrtf(u78(base+1)),a=u78(base+2);
        float fx,fy;V point=face_point(s,target,make_float3(1-t,t*(1-a),t*a),&fx,&fy);
        V d=point-p;V phys={d.x*s.spacing,d.y*s.spacing,d.z};float r2=fmaxf(.0001f,dot3(phys,phys));float dist=sqrtf(r2);V dir=phys*(1/dist);
        V ns=shade_normal(s,target,fx,fy),ng=flat_normal(s,target);float4 source=em[target];
        float cosine=-dot3(ns,dir)*source.z;if(cosine<=0)continue;
        Hit h;if(first_hit(s,p,dir,fmaxf(0.f,dist-.001f),false,h))continue;
        float ql=prob[target]/s.geometry[target].w;float qr=fabsf(dot3(ng,dir))/(4*pi*r2);
        float scale=cosine/(pi*r2*nl*fmaxf(1e-28f,ql));
        float balance=nl*ql/fmaxf(1e-30f,nl*ql+nd*qr);
        lightL+=source.x*scale;lightH+=source.y*scale;
        misL+=source.x*scale*balance;misH+=source.y*scale*balance;
    }
    float *o=res+j*6;o[0]=reverseL;o[1]=reverseH;o[2]=misL;o[3]=misH;o[4]=lightL;o[5]=lightH;
}
extern "C" int refinements78(const float4 *g,const float3 *normals,const float *material,const float3 *primary,int nq,const float3 *spec,int ns,
 const int *ids,int n,const float3 *dirs,int nd,int nl,const float4 *em,const float *prob,const float *cdf,int seed,
 float x,float y,float z,float spacing,float top,int anti,int highQuality,
 float4 *direct,float4 *scratch,float *smooth_spec,float *flat_spec,float *diffuse,cudaStream_t st){
    Scene s{g,normals,material,{x,y,z},spacing,top,anti,1};
    primary_faces<<<(faces+127)/128,128,0,st>>>(s,primary,nq,direct,scratch);
    specular_beams<<<(faces*ns+127)/128,128,0,st>>>(s,spec,ns,smooth_spec);
    if(highQuality){Scene flat=s;flat.smooth=0;specular_beams<<<(faces*ns+127)/128,128,0,st>>>(flat,spec,ns,flat_spec);}
    new_diffuse78<<<(n+127)/128,128,0,st>>>(s,ids,n,dirs,nd,nl,em,prob,cdf,(unsigned)seed,diffuse);
    return cudaGetLastError();
}
