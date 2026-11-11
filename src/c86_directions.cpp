// Enumerate the subset of the organizer's full launch lattice within angular caps.
#include "c66_lattice.cpp"
#include <vector>
#include <cstdlib>
#include <cstring>
extern "C" int directions86(const double *boxes,int count,long long rays,uint32_t **returned,long long *size){
    Basis a{1,0,0},b{1/golden,1./(rays-1),1};reduce(a,b);
    const double det=a.u*b.v-b.u*a.v;
    std::vector<uint32_t> ids;
    try{
        for(int j=0;j<count;j++){
            const double *q=boxes+4*j;
            double i0=1e40,i1=-1e40,j0=1e40,j1=-1e40;
            for(double u:{q[0],q[1]})for(double v:{q[2],q[3]}){
                add_bound((u*b.v-v*b.u)/det,i0,i1);add_bound((v*a.u-u*a.v)/det,j0,j1);
            }
            for(long long i=std::floor(i0)-1;i<=std::ceil(i1)+1;i++)
                for(long long k=std::floor(j0)-1;k<=std::ceil(j1)+1;k++){
                    const double u=i*a.u+k*b.u,v=i*a.v+k*b.v;
                    if(u<q[0]||u>q[1]||v<q[2]||v>q[3])continue;
                    long long index=i*a.index+k*b.index;
                    if(index>=0&&index<rays)ids.push_back(uint32_t(index));
                }
        }
        std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
        *size=ids.size();*returned=(uint32_t*)std::malloc(ids.size()*sizeof(uint32_t));
        if(!*returned&&ids.size())return 1;
        std::memcpy(*returned,ids.data(),ids.size()*sizeof(uint32_t));return 0;
    }catch(...){return 2;}
}
extern "C" void directions86_free(uint32_t *p){std::free(p);}
