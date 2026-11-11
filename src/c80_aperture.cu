// Project the angular footprint of each finite receiver triangle on a dual Fibonacci lattice.
#include <cuda_runtime.h>
#include <cmath>
static constexpr int C=255*255;
static constexpr double tau=6.2831853071795864769;
__device__ inline double3 sub(double3 a,double3 b){return make_double3(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ inline double dot(double3 a,double3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ inline double3 cr(double3 a,double3 b){return make_double3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__device__ inline double2 uv(double3 a){double inv=rsqrt(dot(a,a));return make_double2(atan2(a.y,a.x)/tau,.5*(1-a.z*inv));}
__global__ void aperture(const float *h,const int *face,int n,float tx,float ty,float tz,const double2 *dual,int nd,const double *sizes,int ns,float *out){
    int i=blockIdx.x*blockDim.x+threadIdx.x;int k=blockIdx.y;if(i>=n)return;
    int f=face[i],cell=f%C,x=cell%255,y=cell/255,idx=y*256+x;
    double sp=256./255.;
    int xoff[3]={0,1,int(f<C)},yoff[3]={0,int(f>=C),1};
    if(f>=C){xoff[1]=1;yoff[1]=1;xoff[2]=0;yoff[2]=1;}
    double3 v[3];double2 p[7];double3 center={0,0,0};
    for(int a=0;a<3;++a){
        v[a]=make_double3((x+xoff[a]-tx)*sp,-(y+yoff[a]-ty)*sp,h[idx+xoff[a]+256*yoff[a]]+1.-tz);
        center.x+=v[a].x/3;center.y+=v[a].y/3;center.z+=v[a].z/3;p[a]=uv(v[a]);
    }
    p[6]=uv(center);
    for(int a=0;a<3;++a){int b=(a+1)%3;p[a+3]=uv(make_double3(.5*(v[a].x+v[b].x),.5*(v[a].y+v[b].y),.5*(v[a].z+v[b].z)));}
    for(int a=0;a<7;++a){p[a].x-=nearbyint(p[a].x-p[6].x);}
    double3 nrm=cr(sub(v[1],v[0]),sub(v[2],v[0]));
    double mean=sizes[k]*fabs(dot(nrm,center))/(4*tau*pow(dot(center,center),1.5));
    // Find the projection direction with the smallest cell width in units of lattice lines.
    double best=1e20,empty=0.,cnt=1.,spread=0.;int selected=-1;double3 bt;
    for(int d=0;d<nd;++d){
        double2 g=dual[k*nd+d];double lo=1e40,hi=-1e40;
        for(int a=0;a<7;++a){double t=g.x*p[a].x+g.y*p[a].y;lo=fmin(t,lo);hi=fmax(t,hi);}
        double w=hi-lo;empty=fmax(empty,(ceil(lo)>floor(hi))?1.:0.);
        if(w<best){best=w;selected=d;bt=make_double3(g.x*p[0].x+g.y*p[0].y,g.x*p[1].x+g.y*p[1].y,g.x*p[2].x+g.y*p[2].y);}
    }
    if(best<12 && selected>=0){
        double a=fmin(bt.x,fmin(bt.y,bt.z)),c=fmax(bt.x,fmax(bt.y,bt.z)),b=bt.x+bt.y+bt.z-a-c;
        cnt=0.;
        if(c-a>1e-8){
            for(long long m=(long long)ceil(a);m<=(long long)floor(c);++m){
                cnt+=m<b?2*(m-a)/fmax(1e-16,(b-a)*(c-a)):2*(c-m)/fmax(1e-16,(c-b)*(c-a));
            }
        }
        spread=fmin(8.,1./sqrt(fmax(.01,best)));
    }
    float *d=out+((k*n+i)*8);d[0]=log1pf(float(fmin(1e10,mean)));d[1]=log1pf(float(fmin(1e8,best)));
    d[2]=fmaxf(0.f,1.f-float(best));d[3]=empty;d[4]=fminf(5.f,float(cnt));d[5]=float(spread);
    d[6]=1.f/(1.f+float(mean));d[7]=fminf(6.f,log1pf(float(mean)/fmaxf(.05f,float(best))));
}
extern "C" int aperture80(const float*h,const int*ids,int n,float x,float y,float z,const double2*d,int nd,const double*s,int ns,float*out,cudaStream_t st){
    aperture<<<dim3((n+127)/128,ns),128,0,st>>>(h,ids,n,x,y,z,d,nd,s,ns,out);return cudaGetLastError();
}
