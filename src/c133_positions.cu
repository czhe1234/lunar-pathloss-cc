#include "c67_transport.cu"
__global__ void pos133(Scene s,const float3*dirs,const int*ids,const float*weights,int count,float2*hitxy,float2*rxxy){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count*48)return;
 int e=j%16,k=j/16%3,i=j/48,f=ids[i*3+k];float fx,fy;
 V p=face_point(s,f,make_float3(1.f/3,1.f/3,1.f/3),&fx,&fy);p.z+=1;
 if(e==0&&k==0){
  float x=0,y=0;
  for(int d=0;d<3;++d){float a,b;V v=face_point(s,ids[i*3+d],make_float3(1.f/3,1.f/3,1.f/3),&a,&b);x+=v.x*weights[i*3+d];y+=v.y*weights[i*3+d];}
  rxxy[i]=make_float2(x/127.5f-1,y/127.5f-1);
 }
 Hit hit;V dir=from3(dirs[e]);hitxy[j]=make_float2(0,0);
 if(!first_hit(s,p,dir,500.f,false,hit))return;
 hitxy[j]=make_float2((p.x+dir.x/s.spacing*hit.distance)/127.5f-1,(p.y+dir.y/s.spacing*hit.distance)/127.5f-1);
}
extern "C" int c133_positions(const float4*g,const float3*dirs,const int*ids,const float*w,int count,float top,float2*hitxy,float2*rxxy,cudaStream_t stream){
 Scene s{g,nullptr,nullptr,{0,0,0},256.f/255.f,top,0,0};
 pos133<<<(count*48+127)/128,128,0,stream>>>(s,dirs,ids,w,count,hitxy,rxxy);
 return cudaGetLastError();
}
