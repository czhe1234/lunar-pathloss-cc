// C132: explicit shared diffuse-power transfer and native first/last-leg queries.
// Uses only CC's native triangle intersection and material implementation.
#include "c67_transport.cu"
constexpr int side132=32, nodes132=1024;
__device__ int nodeface132(int j){
 int c=min(254,int((j%32+.5f)*255.f/32.f)),r=min(254,int((j/32+.5f)*255.f/32.f));
 return r*255+c;
}
__device__ int nodeid132(V p){
 int c=min(31,max(0,int(p.x*32.f/255.f))),r=min(31,max(0,int(p.y*32.f/255.f)));
 return r*32+c;
}
__device__ void features132(const Beam& b,V n,V ng,float *out){
 out[0]=log1pf(b.length)/6.f;out[1]=-dot3(b.dir,n);out[2]=-dot3(b.dir,ng);
 out[3]=n.x;out[4]=n.y;out[5]=n.z;out[6]=b.dir.x;out[7]=b.dir.y;out[8]=b.dir.z;
 out[9]=log1pf(float(b.crossings))/3.f;out[10]=logf(fmaxf(1e-5f,fabsf(dot3(b.dir,ng))))/10.f;
 out[11]=dot3(n,ng);
}
__global__ void graph132(Scene s,const float3* dirs,int nd,int *src,int *dst,float2 *weight,float *feat,float *area){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=nodes132*nd)return;
 int i=j/nd,e=j%nd,f=nodeface132(i);float fx,fy;
 V p=face_point(s,f,make_float3(1.f/3,1.f/3,1.f/3),&fx,&fy),n=shade_normal(s,f,fx,fy);
 float ai=64.f/fmaxf(.08f,flat_normal(s,f).z);if(e==0)area[i]=ai;
 V dir=from3(dirs[e]);float mu=dot3(n,dir);dst[j]=i+(mu<0?nodes132:0);
 float *a=feat+j*12;for(int k=0;k<12;++k)a[k]=0;
 src[j]=0;weight[j]=make_float2(0,0);Hit hit;
 // Signed normal offset selects the receiving side; no TX enters this graph.
 V off=n*(mu>=0?.0005f:-.0005f);p=p+V{off.x/s.spacing,off.y/s.spacing,off.z};
 if(!first_hit(s,p,dir,500.f,false,hit))return;
 V q=p+V{dir.x/s.spacing,dir.y/s.spacing,dir.z}*hit.distance;
 int node=nodeid132(q);if(node==i)return;
 V ns=shade_normal(s,hit.face,hit.fx,hit.fy),nf=flat_normal(s,hit.face);
 float co=-dot3(ns,dir);src[j]=node+(co<0?nodes132:0);
 int jf=nodeface132(node);float aj=64.f/fmaxf(.08f,flat_normal(s,jf).z);
 float scale=4.f*fabsf(mu)*fabsf(co)/(nd*fmaxf(.03f,fabsf(dot3(nf,dir))))*ai/aj;
 C ts,tp;float rs,rp;coefficients(s,fabsf(mu),0,ts,tp,rs,rp);float lo=.5f*(rs+rp)*.632f*.632f;
 coefficients(s,fabsf(mu),1,ts,tp,rs,rp);float hi=.5f*(rs+rp)*.632f*.632f;
 weight[j]=make_float2(scale*lo,scale*hi);
 a[0]=log1pf(hit.distance)/6.f;a[1]=mu;a[2]=co;
 a[3]=n.x;a[4]=n.y;a[5]=n.z;a[6]=dir.x;a[7]=dir.y;a[8]=dir.z;
 a[9]=ns.x;a[10]=ns.y;a[11]=ns.z;
}
__global__ void source132(Scene s,const float*area,float2 *q,float*feat){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nodes132)return;
 int f=nodeface132(i);float fx,fy;V p=face_point(s,f,make_float3(1.f/3,1.f/3,1.f/3),&fx,&fy);
 V n=shade_normal(s,f,fx,fy),ng=flat_normal(s,f);Beam b=incident(s,p);
 float v=.632f*.632f*fabsf(dot3(b.dir,ng))/fmaxf(.0625f,b.length*b.length)*area[i];
 int j=i+(-dot3(b.dir,n)<0?nodes132:0);
 q[i]=make_float2(0,0);q[i+nodes132]=make_float2(0,0);
 q[j]=make_float2(v*reflected(s,b,n,0),v*reflected(s,b,n,1));
 features132(b,n,ng,feat+i*12);
}
__global__ void receiver132(Scene s,const float3*dirs,int nd,const int*faces3,const float*interp,
 int count,int *node,float *aw,float2 *q,float *qfeat,float *afeat,float *directfeat){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count*3*nd)return;
 int e=j%nd,k=(j/nd)%3,i=j/(3*nd),f=faces3[i*3+k];
 float fx,fy;V p=face_point(s,f,make_float3(1.f/3,1.f/3,1.f/3),&fx,&fy);p.z+=1;
 if(e==0){Beam b=incident(s,p);features132(b,shade_normal(s,f,fx,fy),flat_normal(s,f),directfeat+(i*3+k)*12);}
 node[j]=0;aw[j]=0;q[j]=make_float2(0,0);
 float *a=afeat+j*12,*bq=qfeat+j*12;
 for(int k=0;k<12;++k){a[k]=0;bq[k]=0;}
 V dir=from3(dirs[e]);Hit hit;if(!first_hit(s,p,dir,500.f,false,hit))return;
 V target=p+V{dir.x/s.spacing,dir.y/s.spacing,dir.z}*hit.distance;
 V n=shade_normal(s,hit.face,hit.fx,hit.fy),ng=flat_normal(s,hit.face);
 float co=-dot3(dir,n);int no=nodeid132(target);node[j]=no+(co<0?nodes132:0);
 float w=interp[i*3+k]*4.f*fabsf(co)/(nd*fmaxf(.03f,fabsf(dot3(ng,dir))));aw[j]=w;
 Beam incoming=incident(s,target);
 features132(incoming,n,ng,bq);
 float sign=-dot3(incoming.dir,n);
 if(sign*co>0){
  float v=.632f*.632f*fabsf(dot3(incoming.dir,ng))/fmaxf(.0625f,incoming.length*incoming.length);
  q[j]=make_float2(v*reflected(s,incoming,n,0),v*reflected(s,incoming,n,1));
 }
 a[0]=log1pf(hit.distance)/6.f;a[1]=co;a[2]=dot3(ng,dir);
 a[3]=n.x;a[4]=n.y;a[5]=n.z;a[6]=dir.x;a[7]=dir.y;a[8]=dir.z;
 a[9]=ng.x;a[10]=ng.y;a[11]=ng.z;
}
extern "C" int c132_graph(const float4*g,const float3*n,const float*mat,const float3*dir,int nd,float top,
 int*src,int*dst,float2*w,float*feat,float*area,cudaStream_t stream){
 Scene s{g,n,mat,{0,0,0},256.f/255.f,top,0,1};
 graph132<<<(nodes132*nd+127)/128,128,0,stream>>>(s,dir,nd,src,dst,w,feat,area);
 return cudaGetLastError();
}
extern "C" int c132_query(const float4*g,const float3*n,const float*mat,const float3*dir,int nd,
 float x,float y,float z,float top,const float*area,const int*ids,const float*interp,int count,
 float2*sq,float*sf,int*rid,float*aw,float2*rq,float*qf,float*af,float*df,cudaStream_t stream){
 Scene s{g,n,mat,{x,y,z},256.f/255.f,top,0,1};
 source132<<<(nodes132+127)/128,128,0,stream>>>(s,area,sq,sf);
 int total=count*3*nd;
 receiver132<<<(total+127)/128,128,0,stream>>>(s,dir,nd,ids,interp,count,rid,aw,rq,qf,af,df);
 return cudaGetLastError();
}
