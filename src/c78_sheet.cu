// Geometry of primary rays at the raised receiver mesh. No labels or additional power terms.
#include "c67_transport.cu"
__global__ void sheet78(Scene s,const int *ids,int n,const float3 *quad,int nq,float *out) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
    int f=ids[j];float wfirst=0,wfront=0,previous=0,grazing=0,logn=0,zero4b=0,zero40m=0,zero4m=0;
    V normal=flat_normal(s,f);
    for(int k=0;k<nq;++k){
        float fx,fy;V p=face_point(s,f,quad[k],&fx,&fy);p.z+=1;
        V dx=p-s.tx;V phys={dx.x*s.spacing,dx.y*s.spacing,dx.z};
        float rr=fmaxf(1e-6f,dot3(phys,phys)),dist=sqrtf(rr);V dir=unit(phys);
        float c=dot3(dir,normal);wfront+=c<=0;grazing+=fabsf(c)<.015f;
        float count=4e9f*s.geometry[f].w*fabsf(c)/(4*3.14159265359f*rr);
        logn+=log1pf(count);zero4b+=expf(-count);zero40m+=expf(-.01f*count);zero4m+=expf(-.001f*count);
        V o=s.tx;o.z-=1;V gd={dir.x/s.spacing,dir.y/s.spacing,dir.z};float left=dist-.0008f;int m=0;
        for(;m<30 && left>0;++m){
            Hit hit;if(!first_hit(s,o,dir,left,false,hit))break;
            float step=hit.distance+.00035f;o=o+gd*step;left-=step;
        }
        wfirst+=m==0;previous+=m;
    }
    float *v=out+j*8;v[0]=wfirst/nq;v[1]=wfront/nq;v[2]=previous/nq;v[3]=grazing/nq;
    v[4]=logn/nq;v[5]=zero4b/nq;v[6]=zero40m/nq;v[7]=zero4m/nq;
}
extern "C" int sheet_geometry78(const float4 *plane,const int *ids,int n,const float3 *quad,int nq,float x,float y,float z,float top,float *out,cudaStream_t st){
    Scene s{plane,nullptr,nullptr,{x,y,z},256.f/255.f,top,0,0};sheet78<<<(n+127)/128,128,0,st>>>(s,ids,n,quad,nq,out);return cudaGetLastError();
}
