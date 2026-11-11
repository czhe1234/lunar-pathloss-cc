// Input-only visibility stress tests and conservative volume certificates for the receiver triangles.
#include "c67_transport.cu"
constexpr int nvariants=13;
struct SourceRange { float x,y,z,original,interpolated; };
__device__ float3 barypoint78(const Scene &s,int face,float3 b) {
    float fx,fy;V p=face_point(s,face,b,&fx,&fy);return make_float3(p.x,p.y,p.z);
}
__global__ void visibility78(const float4 *main,const float4 *anti,const float4 *coarse2,const float4 *coarse4,
 const int *ids0,const int *ids1,const int *pix,int n,const float3 *samples,SourceRange tx,float top,float *out) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=n*3*nvariants)return;
    int j=idx/(3*nvariants),v=(idx/3)%nvariants,f=(v==2?ids1:ids0)[pix[j]*3+idx%3];
    const float4 *g=v==2?anti:(v==11?coarse2:(v==12?coarse4:main));
    float x=tx.x,y=tx.y,z=tx.z,rx=1.f;
    if(v==3)z=tx.original;if(v==4)z=tx.interpolated;
    if(v==5){z-=.5f;rx=.75f;}if(v==6){z+=.5f;rx=1.25f;}
    if(v==7)x-=.5f;if(v==8)x+=.5f;if(v==9)y-=.5f;if(v==10)y+=.5f;
    x=fminf(255,fmaxf(0,x));y=fminf(255,fmaxf(0,y));
    Scene s{g,nullptr,nullptr,{x,y,z},256.f/255.f,top,int(v==2),0};
    // Quadrature arrays are concatenated as 9, 64 and 16 points.
    int nq=v==0?9:((v==1||v==2)?64:16),off=v==0?0:((v==1||v==2)?9:73);
    int clear=0;
    for(int i=0;i<nq;++i){
        float fx,fy;V p=face_point(s,f,samples[off+i],&fx,&fy);p.z+=rx;
        V d=p-s.tx;V world={d.x*s.spacing,d.y*s.spacing,d.z};
        float distance=sqrtf(dot3(world,world));Hit hit;
        bool blocked=first_hit(s,s.tx,unit(world),distance-.0008f,false,hit);
        clear+=!blocked;
    }
    out[idx]=float(clear)/nq;
}
struct Bounds {float xmin,xmax,ymin,ymax;};
__device__ void extrema(float &lo,float &hi,float v){lo=fminf(lo,v);hi=fmaxf(hi,v);}
__device__ void face_range78(float4 a,float xl,float xh,float yl,float yh,bool both,float &lo,float &hi){
    // Exact extrema on an axis-aligned rectangle clipped to a grid square:
    // four corners plus intersections with the two possible grid diagonals.
    float ab=a.y-a.x,ac=a.z-a.x,bd=a.w-a.y,cd=a.w-a.z;
    for(int i=0;i<2;++i)for(int j=0;j<2;++j){
        float x=i?xh:xl,y=j?yh:yl;
        extrema(lo,hi,y<=x?a.x+ab*x+bd*y:a.x+cd*x+ac*y);
        if(both)extrema(lo,hi,x+y<=1?a.x+ab*x+ac*y:a.w+cd*(x-1)+bd*(y-1));
    }
    float l=fmaxf(xl,yl),u=fminf(xh,yh);
    if(l<=u){extrema(lo,hi,a.x+(a.w-a.x)*l);extrema(lo,hi,a.x+(a.w-a.x)*u);}
    if(both){
        l=fmaxf(xl,1-yh);u=fminf(xh,1-yl);
        if(l<=u){extrema(lo,hi,a.z+(a.y-a.z)*l);extrema(lo,hi,a.z+(a.y-a.z)*u);}
    }
}
__device__ bool terrain_range78(const float4 *corners,Bounds b,bool both,float limit,float &lo,float &hi){
    b.xmin=fmaxf(0,b.xmin);b.xmax=fminf(255,b.xmax);b.ymin=fmaxf(0,b.ymin);b.ymax=fminf(255,b.ymax);
    int x0=min(254,(int)b.xmin),x1=min(254,(int)b.xmax),y0=min(254,(int)b.ymin),y1=min(254,(int)b.ymax);
    lo=1e30f;hi=-1e30f;
    for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x){
        float4 q=corners[y*255+x];
        float quick=fmaxf(fmaxf(q.x,q.y),fmaxf(q.z,q.w));
        if(limit<1e20f && quick<limit)continue;
        face_range78(q,fmaxf(0,b.xmin-x),fminf(1,b.xmax-x),fmaxf(0,b.ymin-y),fminf(1,b.ymax-y),both,lo,hi);
        if(hi>=limit)return true;
    }
    return false;
}
__device__ float interp78(float x,float y,float t){return x+(y-x)*t;}
__device__ float far78(float a,float b,float c,float d){return fmaxf(fabsf(a-d),fabsf(b-c));}
__device__ float near78(float a,float b,float c,float d){return fmaxf(0,fmaxf(a-d,c-b));}
__global__ void certificates78(const float4 *main,const float4 *anti,const float4 *corners,const int *ids0,const int *ids1,
 const float *w0,const float *w1,const int *pix,int n,SourceRange tx,const float3 *subtris,float4 *out){
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=n*3*2*3*4)return;
    int sub=id%4,part=(id/4)%3,side=(id/12)%2,mode=(id/24)%3,row=id/72;
    bool both=mode!=0;bool narrowwide=mode==1; // mode1 allows both diagonals and competing TX altitude, mode2 adds coordinate/height tolerances.
    int f=(side?ids1:ids0)[3*pix[row]+part];float w=(side?w1:w0)[3*pix[row]+part];
    if((!both&&side)||w<=0.f){out[id]=make_float4(1,0,1e3f,0);return;}
    Scene s{side?anti:main,nullptr,nullptr,{tx.x,tx.y,tx.z},256.f/255.f,0,side,0};
    Bounds rx={1e5f,-1e5f,1e5f,-1e5f};float zl=1e5f,zu=-1e5f;
    for(int k=0;k<3;++k){
        float3 p=barypoint78(s,f,subtris[sub*3+k]);
        extrema(rx.xmin,rx.xmax,p.x);extrema(rx.ymin,rx.ymax,p.y);extrema(zl,zu,p.z);
    }
    float xy=mode==2?.5f:0.f,dz=mode==2?.5f:0.f;
    Bounds st={fmaxf(0,tx.x-xy),fminf(255,tx.x+xy),fmaxf(0,tx.y-xy),fminf(255,tx.y+xy)};
    float tz0=both?fminf(tx.z,fminf(tx.original,tx.interpolated))-dz:tx.z;
    float tz1=both?fmaxf(tx.z,fmaxf(tx.original,tx.interpolated))+dz:tx.z;
    if(mode==2){
        rx={fmaxf(0,rx.xmin-xy),fminf(255,rx.xmax+xy),fmaxf(0,rx.ymin-xy),fminf(255,rx.ymax+xy)};
        terrain_range78(corners,rx,true,1e30f,zl,zu);
    }
    // For mode1, side-specific receiving triangles are paired with their native heights.
    zl+=mode==2?.75f:1.f;zu+=mode==2?1.25f:1.f;
    float dx=far78(rx.xmin,rx.xmax,st.xmin,st.xmax),dy=far78(rx.ymin,rx.ymax,st.ymin,st.ymax);
    float dd=far78(zl,zu,tz0,tz1),far=sqrtf((dx*dx+dy*dy)*s.spacing*s.spacing+dd*dd);
    float nx=near78(rx.xmin,rx.xmax,st.xmin,st.xmax),ny=near78(rx.ymin,rx.ymax,st.ymin,st.ymax),nz=near78(zl,zu,tz0,tz1);
    float near=sqrtf((nx*nx+ny*ny)*s.spacing*s.spacing+nz*nz);
    int segments=max(1,int(ceilf(2*fmaxf(dx,dy))));bool safe=true;
    // The union of every ray in each parameter interval is contained inside this swept box.
    for(int k=segments-1;k>=0;--k){
        float t0=float(k)/segments,t1=float(k+1)/segments;
        Bounds b={fminf(interp78(st.xmin,rx.xmin,t0),interp78(st.xmin,rx.xmin,t1))-1e-5f,
                  fmaxf(interp78(st.xmax,rx.xmax,t0),interp78(st.xmax,rx.xmax,t1))+1e-5f,
                  fminf(interp78(st.ymin,rx.ymin,t0),interp78(st.ymin,rx.ymin,t1))-1e-5f,
                  fmaxf(interp78(st.ymax,rx.ymax,t0),interp78(st.ymax,rx.ymax,t1))+1e-5f};
        float lower=fminf(interp78(tz0,zl,t0),interp78(tz0,zl,t1))-.0002f;
        float low,high;if(terrain_range78(corners,b,both,lower,low,high)){safe=false;break;}
    }
    out[id]=make_float4(safe?1.f:0.f,far,near,0);
}
extern "C" int geometry78(const float4 *g0,const float4 *g1,const float4 *g2,const float4 *g4,const float4 *corners,const int *i0,const int *i1,
 const float *w0,const float *w1,const int *pix,int n,const float3 *qs,const float3 *sub,float x,float y,float z,float zo,float zi,float top,
 float *v,float4 *c,cudaStream_t st){
    SourceRange tx{x,y,z,zo,zi};
    visibility78<<<(n*3*nvariants+127)/128,128,0,st>>>(g0,g1,g2,g4,i0,i1,pix,n,qs,tx,top,v);
    certificates78<<<(n*72+127)/128,128,0,st>>>(g0,g1,corners,i0,i1,w0,w1,pix,n,tx,sub,c);
    return cudaGetLastError();
}
