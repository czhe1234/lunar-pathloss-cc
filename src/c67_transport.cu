// Independent one-bounce transport on a candidate triangulated height field.
#include <cuda_runtime.h>
#include <cmath>
struct V {
    float x,y,z;
    __device__ V operator+(V b) const { return {x+b.x,y+b.y,z+b.z}; }
    __device__ V operator-(V b) const { return {x-b.x,y-b.y,z-b.z}; }
    __device__ V operator*(float s) const { return {x*s,y*s,z*s}; }
};
__device__ float dot3(V a,V b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ V cross3(V a,V b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
__device__ V unit(V a) { return a*rsqrtf(fmaxf(1e-20f,dot3(a,a))); }
__device__ V from3(float3 a) { return {a.x,a.y,a.z}; }
constexpr int cells=255*255, faces=2*cells;
struct Scene {
    const float4 *geometry;   // z at cell origin, dz/dcolumn, dz/drow, triangular area
    const float3 *normals;    // angle-weighted unit vertex normals
    const float *material;    // per band, per cosine, Ts, Tp complex, Rs/Rp intensity
    V tx;
    float spacing, top;
    int anti, smooth;
};
struct Hit { int face; float distance,fx,fy; bool receiver; };
__device__ void barycentric(const Scene &s,int face,float fx,float fy,int ids[3],float weights[3]) {
    int cell=face%cells, a=(cell/255)*256+cell%255;
    if(!s.anti && face<cells) {
        ids[0]=a;ids[1]=a+1;ids[2]=a+257;
        weights[0]=1-fx;weights[1]=fx-fy;weights[2]=fy;
    } else if(!s.anti) {
        ids[0]=a;ids[1]=a+257;ids[2]=a+256;
        weights[0]=1-fy;weights[1]=fx;weights[2]=fy-fx;
    } else if(face<cells) {
        ids[0]=a;ids[1]=a+1;ids[2]=a+256;
        weights[0]=1-fx-fy;weights[1]=fx;weights[2]=fy;
    } else {
        ids[0]=a+257;ids[1]=a+256;ids[2]=a+1;
        weights[0]=fx+fy-1;weights[1]=1-fx;weights[2]=1-fy;
    }
}
__device__ V flat_normal(const Scene &s,int f) {
    float4 geom=s.geometry[f];return unit({-geom.y/s.spacing,-geom.z/s.spacing,1});
}
__device__ V shade_normal(const Scene &s,int face,float fx,float fy) {
    if(!s.smooth) return flat_normal(s,face);
    int ids[3];float w[3];barycentric(s,face,fx,fy,ids,w);
    V n={0,0,0};
    #pragma unroll
    for(int i=0;i<3;++i) n=n+from3(s.normals[ids[i]])*w[i];
    return unit(n);
}
__device__ V face_point(const Scene &s,int face,float3 bary,float *fx,float *fy) {
    if(!s.anti && face<cells) { *fx=bary.y+bary.z; *fy=bary.z; }
    else if(!s.anti) { *fx=bary.y; *fy=bary.y+bary.z; }
    else if(face<cells) { *fx=bary.y; *fy=bary.z; }
    else { *fx=bary.x+bary.z; *fy=bary.x+bary.y; }
    int cell=face%cells;float4 g=s.geometry[face];
    return {cell%255+*fx,cell/255+*fy,g.x+g.y*(*fx)+g.z*(*fy)};
}
__device__ bool first_hit(const Scene &s,V o,V dir,float limit,bool receivers,Hit &answer) {
    const float eps=.00015f;
    V g={dir.x/s.spacing,dir.y/s.spacing,dir.z};
    V p=o+g*eps;
    if(p.x<0||p.x>255||p.y<0||p.y>255) return false;
    int ix=min(254,(int)floorf(p.x)),iy=min(254,(int)floorf(p.y));
    int dx=g.x>=0?1:-1,dy=g.y>=0?1:-1;
    float stepX=fabsf(g.x)>1e-10f?fabsf(1/g.x):1e25f;
    float stepY=fabsf(g.y)>1e-10f?fabsf(1/g.y):1e25f;
    float nextX=stepX<1e24f?(ix+(dx>0)-o.x)/g.x:1e25f;
    float nextY=stepY<1e24f?(iy+(dy>0)-o.y)/g.y:1e25f;
    float enter=eps;
    for(int iter=0;iter<515 && enter<limit;++iter) {
        if(ix<0||ix>=255||iy<0||iy>=255) break;
        float z=o.z+enter*g.z;
        if((z>s.top+1.001f && g.z>=0)||(z<-.001f && g.z<=0)) break;
        float end=fminf(limit,fminf(nextX,nextY));
        float best=limit+1;
        int cell=iy*255+ix;
        #pragma unroll
        for(int facePart=0;facePart<2;++facePart) {
            int f=cell+facePart*cells; float4 plane=s.geometry[f];
            float den=g.z-plane.y*g.x-plane.z*g.y;
            if(fabsf(den)<1e-9f) continue;
            float num=plane.x+plane.y*(o.x-ix)+plane.z*(o.y-iy)-o.z;
            for(int layer=0;layer<=(receivers?1:0);++layer) {
                float t=(num+layer)/den;
                if(t<enter || t>end || t>=best || t>=limit) continue;
                float fx=fmaf(t,g.x,o.x)-ix,fy=fmaf(t,g.y,o.y)-iy;
                int part=s.anti?(fx+fy>1):(fy>fx);
                if(part!=facePart) continue;
                best=t; answer={f,t,fx,fy,bool(layer)};
            }
        }
        if(best<=end) return true;
        if(end>=limit) break;
        bool mx=nextX<=nextY,my=nextY<=nextX;
        if(mx){ix+=dx;nextX+=stepX;}
        if(my){iy+=dy;nextY+=stepY;}
        enter=fmaxf(enter,end);
    }
    return false;
}
struct C { float re,im;
    __device__ C operator*(C b)const{return {re*b.re-im*b.im,re*b.im+im*b.re};}
    __device__ C operator*(float a)const{return {re*a,im*a};}
    __device__ C operator+(C b)const{return {re+b.re,im+b.im};}
    __device__ float power()const{return re*re+im*im;}
};
struct Pol { C h,v; };
__device__ void coefficients(const Scene &s,float cosine,int band,C &ts,C &tp,float &rs,float &rp) {
    float u=fminf(4095.9999f,fmaxf(0.f,cosine*4096));int j=min(4095,int(u));float a=u-j;
    const float *row=s.material+(band*4097+j)*6;
    float r[6];
    #pragma unroll
    for(int k=0;k<6;++k)r[k]=fmaf(a,row[k+6]-row[k],row[k]);
    ts={r[0],r[1]};tp={r[2],r[3]};rs=r[4];rp=r[5];
}
__device__ void project(V k,V n,V h,V v,float &a,float &b) {
    V te=cross3(k,n);float d=dot3(te,te);
    if(d<1e-12f){a=1;b=0;}else{te=te*rsqrtf(d);a=dot3(te,h);b=dot3(te,v);}
}
struct Beam { V dir,h,v; float length; int crossings; Pol pol[2]; };
__device__ Beam incident(const Scene &s,V endpoint) {
    V delta=endpoint-s.tx;
    V physical={delta.x*s.spacing,delta.y*s.spacing,delta.z};
    Beam b;b.length=sqrtf(fmaxf(1e-8f,dot3(physical,physical)));b.dir=physical*(1/b.length);
    b.h=unit({-b.dir.y,b.dir.x,0});
    if(dot3(b.h,b.h)<.5f)b.h={0,1,0};
    b.v=cross3(b.h,b.dir);b.crossings=0;
    b.pol[0]={{0,0},{1,0}};b.pol[1]=b.pol[0];
    V origin=s.tx;V gd={b.dir.x/s.spacing,b.dir.y/s.spacing,b.dir.z};float remaining=b.length-.0008f;
    for(int j=0;j<32;++j) {
        Hit hit;if(!first_hit(s,origin,b.dir,remaining,false,hit))break;
        V n=shade_normal(s,hit.face,hit.fx,hit.fy);
        float a,d;project(b.dir,n,b.h,b.v,a,d);
        #pragma unroll
        for(int f=0;f<2;++f) {
            C ts,tp;float rs,rp;coefficients(s,fabsf(dot3(b.dir,n)),f,ts,tp,rs,rp);
            C te=(b.pol[f].h*a+b.pol[f].v*d)*ts;
            C tm=(b.pol[f].h*(-d)+b.pol[f].v*a)*tp;
            b.pol[f]={te*a+tm*(-d),te*d+tm*a};
        }
        ++b.crossings;float march=hit.distance+.00035f;origin=origin+gd*march;remaining-=march;
        if(remaining<=0)break;
    }
    return b;
}
__device__ float reflected(const Scene &s,const Beam &beam,V n,int f) {
    float a,b;project(beam.dir,n,beam.h,beam.v,a,b);
    C ts,tp;float rs,rp;coefficients(s,fabsf(dot3(beam.dir,n)),f,ts,tp,rs,rp);
    return rs*(beam.pol[f].h*a+beam.pol[f].v*b).power()+rp*(beam.pol[f].h*(-b)+beam.pol[f].v*a).power();
}
__global__ void primary_faces(Scene s,const float3 *bary,int ns,float4 *direct,float4 *emission) {
    int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=faces)return;
    float low=0,high=0,los=0,ncross=0;
    for(int j=0;j<ns;++j) {
        float x,y;V p=face_point(s,face,bary[j],&x,&y);p.z+=1;Beam b=incident(s,p);
        float inv=1/fmaxf(.0625f,b.length*b.length);
        low+=(b.pol[0].h.power()+b.pol[0].v.power())*inv;
        high+=(b.pol[1].h.power()+b.pol[1].v.power())*inv;
        los+=b.crossings==0;ncross+=b.crossings;
    }
    direct[face]=make_float4(low/ns,high/ns,los/ns,ncross/ns);
    float x,y;V p=face_point(s,face,make_float3(1.f/3,1.f/3,1.f/3),&x,&y);Beam b=incident(s,p);
    V n=shade_normal(s,face,x,y),ng=flat_normal(s,face);
    float ci=-dot3(b.dir,ng),w=.632f*.632f*fabsf(ci)/fmaxf(.0625f,b.length*b.length);
    emission[face]=make_float4(w*reflected(s,b,n,0),w*reflected(s,b,n,1),-dot3(b.dir,n)>=0?1.f:-1.f,ci);
}
__global__ void specular_beams(Scene s,const float3 *bary,int ns,float *received) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=faces*ns)return;
    int face=id/ns,j=id%ns;float x,y;V p=face_point(s,face,bary[j],&x,&y);
    Beam b=incident(s,p);V n=shade_normal(s,face,x,y),ng=flat_normal(s,face);
    V outgoing=b.dir-n*(2*dot3(b.dir,n));
    float weight=(1-.632f*.632f)*s.geometry[face].w*fabsf(dot3(b.dir,ng))/(ns*fmaxf(.0625f,b.length*b.length));
    float low=weight*reflected(s,b,n,0),high=weight*reflected(s,b,n,1);
    if(fmaxf(low,high)<1e-24f)return;
    V gd={outgoing.x/s.spacing,outgoing.y/s.spacing,outgoing.z};
    for(int bounce=0;bounce<20;++bounce) {
        Hit hit;if(!first_hit(s,p,outgoing,1e5f,true,hit)||!hit.receiver)break;
        float den=s.geometry[hit.face].w*fmaxf(1e-7f,fabsf(dot3(outgoing,flat_normal(s,hit.face))));
        atomicAdd(received+hit.face*3,low/den);
        atomicAdd(received+hit.face*3+1,high/den);
        atomicAdd(received+hit.face*3+2,1.f/ns);
        p=p+gd*(hit.distance+.00035f);
    }
}
__global__ void diffuse_receivers(Scene s,const float3 *directions,int count,const float4 *sources,float4 *output) {
    int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=faces)return;
    float x,y;V p=face_point(s,face,make_float3(1.f/3,1.f/3,1.f/3),&x,&y);p.z+=1;
    float lo=0,hi=0,lo2=0,hi2=0;
    for(int i=0;i<count;++i) {
        V dir=from3(directions[i]);Hit hit;if(!first_hit(s,p,dir,1e5f,false,hit))continue;
        float4 source=sources[hit.face];
        V ng=flat_normal(s,hit.face),ns=shade_normal(s,hit.face,hit.fx,hit.fy);
        float cosine=-dot3(ns,dir)*source.z;
        if(cosine<=0)continue;
        float scale=4.f*cosine/(count*fmaxf(.0001f,fabsf(dot3(ng,dir))));
        float l=source.x*scale,h=source.y*scale;
        lo+=l;hi+=h;lo2+=l*l;hi2+=h*h;
    }
    output[face]=make_float4(lo,hi,lo2,hi2);
}
extern "C" int transport(const float4 *geometry,const float3 *normals,const float *material,
                const float3 *primarySamples,int nq,const float3 *specSamples,int ns,
                const float3 *directions,int nd,float x,float y,float z,float spacing,float top,
                int anti,int smooth,float4 *direct,float4 *emission,float *spec,float4 *diffuse,
                cudaStream_t stream) {
    Scene scene{geometry,normals,material,{x,y,z},spacing,top,anti,smooth};
    primary_faces<<<(faces+127)/128,128,0,stream>>>(scene,primarySamples,nq,direct,emission);
    specular_beams<<<(faces*ns+127)/128,128,0,stream>>>(scene,specSamples,ns,spec);
    diffuse_receivers<<<(faces+127)/128,128,0,stream>>>(scene,directions,nd,emission,diffuse);
    return cudaGetLastError();
}
