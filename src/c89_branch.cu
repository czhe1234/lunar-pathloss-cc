// Frequency-specific survival of transmitted rays through a triangulated receiver scene.
#include "c67_transport.cu"
constexpr int F89=17;
__global__ void branch89(Scene s,const int *ids,int n,const float3 *quad,int nq,float *out){
    int u=blockIdx.x*blockDim.x+threadIdx.x;if(u>=n)return;
    int face=ids[u];V ng=flat_normal(s,face);float v[F89]={0};
    for(int j=0;j<nq;++j){
        float fx,fy;V ep=face_point(s,face,quad[j],&fx,&fy);ep.z+=1;
        V off=ep-s.tx;V dv={off.x*s.spacing,off.y*s.spacing,off.z};
        float len=sqrtf(fmaxf(.0625f,dot3(dv,dv)));V dir=dv*(1.f/len);
        V h=unit({-dir.y,dir.x,0});if(dot3(h,h)<.5f)h={0,1,0};V ve=cross3(h,dir);
        V step={dir.x/s.spacing,dir.y/s.spacing,dir.z};V pos=s.tx;float rem=len-.0008f;
        Pol pol[2]={{{0,0},{1,0}},{{0,0},{1,0}}};float pt[2]={1,1};
        int crossings=0;
        for(int k=0;k<100;++k){
            Hit hit;if(!first_hit(s,pos,dir,rem,false,hit))break;
            V normal=shade_normal(s,hit.face,hit.fx,hit.fy);float a,b;project(dir,normal,h,ve,a,b);
            for(int band=0;band<2;++band){
                C ts,tp;float rs,rp;coefficients(s,fabsf(dot3(dir,normal)),band,ts,tp,rs,rp);
                float transmit=ts.power()+tp.power(),reflect=rs+rp;
                pt[band]*=transmit/fmaxf(1e-20f,transmit+reflect);
                C te=(pol[band].h*a+pol[band].v*b)*ts;
                C tm=(pol[band].h*(-b)+pol[band].v*a)*tp;
                pol[band]={te*a+tm*(-b),te*b+tm*a};
            }
            ++crossings;float mv=hit.distance+.00035f;pos=pos+step*mv;rem-=mv;if(rem<.0002f)break;
        }
        float incident=fabsf(dot3(ng,dir));
        float rays=4e9f/(4.f*3.14159265f)*s.geometry[face].w*incident/fmaxf(.0625f,len*len);
        v[0]+=rays;v[1]+=crossings;v[2]+=(crossings==0);v[3]+=incident;v[4]+=len;
        for(int b=0;b<2;++b){
            float power=pol[b].h.power()+pol[b].v.power();int ix=5+6*b;
            v[ix]+=power;
            v[ix+1]+=rays*pt[b]; // Expected transmitted rays per complete face and per seed
            v[ix+2]+=pt[b];
            v[ix+3]+=power*power/fmaxf(1e-24f,pt[b]*fmaxf(1e-6f,rays)); // relative compound-shot variance
            v[ix+4]+=rays*pt[b]*(crossings>0);
            v[ix+5]+=power*(crossings>0);
        }
    }
    for(int i=0;i<F89;++i)out[u*F89+i]=v[i]/nq;
}
extern "C" int sample89(const float4 *g,const float3 *normals,const float *material,const int *ids,int n,
    const float3 *q,int nq,float x,float y,float z,float top,float *out,cudaStream_t st){
    Scene s{g,normals,material,{x,y,z},256.f/255.f,top,0,1};
    branch89<<<(n+127)/128,128,0,st>>>(s,ids,n,q,nq,out);
    return cudaGetLastError();
}
