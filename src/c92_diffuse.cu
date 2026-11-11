// Two-sided native-facet transport with direction sampling and illuminated-face sampling.
#include "c78_power.cu"
__global__ void seed92(const float4 *em,float4 *field){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=faces)return;auto e=em[i];
    field[i]=e.z>=0?make_float4(e.x,e.y,0,0):make_float4(0,0,e.x,e.y);
}
__device__ void gather92(const Scene&s,bool ground,V n,V dir,const float4 *field,int src,float fx,float fy,float scale,
                         float4 &result){
    V ns=shade_normal(s,src,fx,fy);float emit=-dot3(ns,dir);float4 v=field[src];
    float lo=emit>=0?v.x:v.z,hi=emit>=0?v.y:v.w;scale*=fabsf(emit);
    if(!ground){result.x+=lo*scale;result.y+=hi*scale;return;}
    float mu=dot3(n,dir);bool side=mu>=0;mu=fminf(1.f,fabsf(mu));
    C ts,tp;float rs,rp;coefficients(s,mu,0,ts,tp,rs,rp);float lr=.5f*(rs+rp);
    coefficients(s,mu,1,ts,tp,rs,rp);float hr=.5f*(rs+rp);
    float weight=scale*mu*.632f*.632f;
    if(side){result.x+=lo*weight*lr;result.y+=hi*weight*hr;}
    else{result.z+=lo*weight*lr;result.w+=hi*weight*hr;}
}
__global__ void propagate92(Scene s,const int *ids,int count,bool ground,const float3 *dirs,int nd,int nl,
   const float4 *field,const float *prob,const float *cdf,unsigned seed,float4 *out){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count)return;int f=ground?j:ids[j];
    float fx,fy;V p=face_point(s,f,make_float3(1.f/3,1.f/3,1.f/3),&fx,&fy);
    if(!ground)p.z+=1;
    V own=shade_normal(s,f,fx,fy);float4 res=make_float4(0,0,0,0);
    const float pi=3.14159265358979323846f;
    float rot=2*pi*u78(seed^mix78(f));float cs=cosf(rot),sn=sinf(rot);
    for(int i=0;i<nd;++i){
        V u=from3(dirs[i]);V dir={u.x*cs-u.y*sn,u.y*cs+u.x*sn,u.z};
        Hit h;if(!first_hit(s,p,dir,500.f,false,h)||(ground&&h.face==f))continue;
        V ng=flat_normal(s,h.face);float cosine=fmaxf(1e-5f,fabsf(dot3(ng,dir)));float r2=fmaxf(.0001f,h.distance*h.distance);
        float qr=cosine/(4*pi*r2),ql=prob[h.face]/s.geometry[h.face].w;
        float bal=nd*qr/fmaxf(1e-28f,nd*qr+nl*ql);
        gather92(s,ground,own,dir,field,h.face,h.fx,h.fy,4*bal/(nd*cosine),res);
    }
    for(int i=0;i<nl;++i){
        unsigned r=seed^mix78(f*7919U+i*8191U);int target=sample78(cdf,(i+u78(r))/nl);
        if(ground&&target==f)continue;
        float t=sqrtf(u78(r+1)),a=u78(r+2),x,y;
        V q=face_point(s,target,make_float3(1-t,t*(1-a),t*a),&x,&y);
        V d=q-p;V phys={d.x*s.spacing,d.y*s.spacing,d.z};float r2=fmaxf(.0001f,dot3(phys,phys));
        float length=sqrtf(r2);V dir=phys*(1/length);Hit h;
        if(first_hit(s,p,dir,length-.001f,false,h))continue;
        V ng=flat_normal(s,target);float ql=prob[target]/s.geometry[target].w,qr=fabsf(dot3(ng,dir))/(4*pi*r2);
        float bal=nl*ql/fmaxf(1e-28f,nl*ql+nd*qr);
        gather92(s,ground,own,dir,field,target,x,y,bal/(pi*r2*nl*fmaxf(1e-28f,ql)),res);
    }
    out[j]=res;
}
extern "C" int initial92(const float4*g,const float3*n,const float*mat,const float3*q,float x,float y,float z,float top,
 float4*direct,float4*em,float4*field,cudaStream_t st){
    Scene s{g,n,mat,{x,y,z},256.f/255.f,top,0,1};
    primary_faces<<<(faces+127)/128,128,0,st>>>(s,q,1,direct,em);
    seed92<<<(faces+127)/128,128,0,st>>>(em,field);return cudaGetLastError();
}
extern "C" int exchange92(const float4*g,const float3*n,const float*mat,const int*ids,int count,int ground,
 const float3*dirs,int nd,int nl,const float4*field,const float *prob,const float*cdf,int seed,float top,float4*out,cudaStream_t st){
    Scene s{g,n,mat,{0,0,0},256.f/255.f,top,0,1};
    propagate92<<<(count+63)/64,64,0,st>>>(s,ids,count,bool(ground),dirs,nd,nl,field,prob,cdf,(unsigned)seed,out);
    return cudaGetLastError();
}
