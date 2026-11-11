// Enumerate a Fibonacci sphere lattice in the angular footprint of a triangle.
// Input vertices are in world axes relative to TX. No labels or masks enter.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
namespace {
constexpr double pi = 3.14159265358979323846;
constexpr double golden = 1.6180339887498948482;
struct Vec {
    double x,y,z;
    Vec operator+(Vec b) const {return {x+b.x,y+b.y,z+b.z};}
    Vec operator-(Vec b) const {return {x-b.x,y-b.y,z-b.z};}
    Vec operator*(double a) const {return {x*a,y*a,z*a};}
};
double dot(Vec a, Vec b) {return a.x*b.x+a.y*b.y+a.z*b.z;}
Vec cross(Vec a, Vec b) {return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
double length(Vec v) {return std::sqrt(dot(v,v));}
struct Basis {double u,v; long long index;};
double scalar(Basis a, Basis b) {return a.u*b.u+a.v*b.v;}
void reduce(Basis &a,Basis &b) {
    for(int iter=0;iter<90;++iter) {
        if(scalar(b,b)<scalar(a,a)) std::swap(a,b);
        long long k=std::llround(scalar(a,b)/scalar(a,a));
        if(k==0) return;
        b={b.u-k*a.u,b.v-k*a.v,b.index-k*a.index};
    }
}
double coord_v(Vec r) {return .5*(1-r.z/length(r));}
void add_bound(double value,double &low,double &high) {
    low=std::min(low,value); high=std::max(high,value);
}
}
extern "C" void count_faces(const double *vertices, int size, long long rays,
                             double cutoff, double *result) {
    Basis a{1,0,0},b{1/golden,1.0/(rays-1),1}; reduce(a,b);
    const double determinant=a.u*b.v-a.v*b.u;
    #pragma omp parallel for schedule(dynamic,48)
    for(int face=0;face<size;++face) {
        Vec v[3];
        for(int k=0;k<3;++k) {
            const double *p=vertices+9*face+3*k; v[k]={p[0],p[1],p[2]};
        }
        Vec center=(v[0]+v[1]+v[2])*(1.0/3);
        Vec normal=cross(v[1]-v[0],v[2]-v[0]);
        double twice_area=length(normal); normal=normal*(1/twice_area);
        Vec sides[3]={cross(v[0],v[1]),cross(v[1],v[2]),cross(v[2],v[0])};
        double sign=dot(sides[0],v[2])>=0 ? 1 : -1;
        double l0=length(v[0]),l1=length(v[1]),l2=length(v[2]);
        double solid=2*std::atan2(std::abs(dot(sides[0],v[2])),l0*l1*l2+dot(v[0],v[1])*l2+dot(v[1],v[2])*l0+dot(v[2],v[0])*l1);
        double *out=result+3*face;
        out[0]=1; out[1]=rays*solid/(4*pi); out[2]=-1;
        if(out[1]>cutoff || std::hypot(center.x,center.y)<3) continue;
        double u0=std::atan2(center.y,center.x)/(2*pi), umin=u0,umax=u0;
        double vmin=coord_v(v[0]), vmax=vmin;
        for(int k=0;k<3;++k) {
            double u=std::atan2(v[k].y,v[k].x)/(2*pi);
            u-=std::round(u-u0); add_bound(u,umin,umax);
            add_bound(coord_v(v[k]),vmin,vmax);
            Vec d=v[(k+1)%3]-v[k];
            double aa=dot(v[k],v[k]), bb=dot(v[k],d), cc=dot(d,d);
            double den=d.z*bb-v[k].z*cc;
            if(den!=0) {
                double t=(v[k].z*bb-d.z*aa)/den;
                if(t>0 && t<1) add_bound(coord_v(v[k]+d*t),vmin,vmax);
            }
        }
        // Allow for the float32 warp used by the original renderer.
        umin-=1e-6; umax+=1e-6; vmin-=1e-6; vmax+=1e-6;
        double mmin=1e30,mmax=-1e30,nmin=1e30,nmax=-1e30;
        for(double u:{umin,umax}) for(double vv:{vmin,vmax}) {
            add_bound((u*b.v-vv*b.u)/determinant,mmin,mmax);
            add_bound((vv*a.u-u*a.v)/determinant,nmin,nmax);
        }
        long long mlo=std::floor(mmin)-1,mhi=std::ceil(mmax)+1;
        long long nlo=std::floor(nmin)-1,nhi=std::ceil(nmax)+1;
        if((mhi-mlo+1)*(nhi-nlo+1)>400000) {out[2]=-2;continue;}
        double scale=(4*pi/rays)*2*dot(center,center)/twice_area, sum=0;
        int hits=0;
        for(long long m=mlo;m<=mhi;++m) for(long long n=nlo;n<=nhi;++n) {
            double u=m*a.u+n*b.u, w=m*a.v+n*b.v;
            if(u<umin || u>umax || w<vmin || w>vmax) continue;
            long long idx=m*a.index+n*b.index;
            if(idx<0 || idx>=rays) continue;
            double az=idx/golden; float phi=float(2*pi)*float(az-std::floor(az));
            float z=1.f-2.f*float(double(idx)/(rays-1));
            float radius=std::sqrt(std::max(0.f,1-z*z));
            Vec dir{radius*std::cos(phi),radius*std::sin(phi),z};
            if(sign*dot(sides[0],dir)<0 || sign*dot(sides[1],dir)<0 || sign*dot(sides[2],dir)<0) continue;
            sum+=1/std::max(1e-12,std::abs(dot(normal,dir))); ++hits;
        }
        out[0]=scale*sum; out[2]=hits;
    }
}
