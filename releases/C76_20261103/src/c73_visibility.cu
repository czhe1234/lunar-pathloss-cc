// Geometry graph of visible pieces of terrain. No material or power in these edges.
#include "c67_transport.cu"
constexpr int side73=64, nodes73=side73*side73, edges73=40;
__device__ int sample_ground73(const Scene &s,float x,float y,float &z) {
    int c=min(254,max(0,int(floorf(x)))),r=min(254,max(0,int(floorf(y))));
    float fx=x-c,fy=y-r;
    int f=r*255+c+(fy>fx)*cells;float4 g=s.geometry[f];
    z=fmaf(g.y,fx,fmaf(g.z,fy,g.x));return f;
}
__device__ V physical73(V d,float spacing){return {d.x*spacing,d.y*spacing,d.z};}
__global__ void edges_kernel73(const float4 *geometry,const float3 *transmitter,const float *top,
                               int count,int *indices,float4 *attributes) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=count*2*nodes73*edges73)return;
    int e=id%edges73,node=(id/edges73)%nodes73,layer=(id/(edges73*nodes73))%2;
    int b=id/(edges73*nodes73*2);float spacing=256.f/255.f;
    Scene s{geometry+b*faces,nullptr,nullptr,from3(transmitter[b]),spacing,top[b],0,0};
    float scale=(255.f-2.f/3.f)/255.f;
    float x=1.f/3.f+(4*(node%64)+1.5f)*scale,y=1.f/3.f+(4*(node/64)+1.5f)*scale,h;
    int own=sample_ground73(s,x,y,h);V p={x,y,h+(layer?1.f:.06f)};
    int target=-1;float dst=0;V q{},dir{};
    if(e<24){
        int ring=e/8;float angle=6.28318530718f*((e%8)+.3333333f*ring)/8.f;
        float ux=cosf(angle),uy=sinf(angle);
        float4 g=s.geometry[own];
        float slope=(g.y*ux+g.z*uy)/spacing;
        float elevation=layer?(ring==0?-.42f:(ring==1?-.09f:.035f)):(ring==0?.014f:(ring==1?.09f:.30f));
        dir=unit({ux,uy,slope+elevation});
        Hit hit;
        if(first_hit(s,p,dir,380.f,false,hit)){
            dst=hit.distance;q=p+V{dir.x/spacing,dir.y/spacing,dir.z}*dst;target=hit.face;
        }
    }else{
        if(e<36){
            int j=e-24;float radius=(j/4==0?2.f:(j/4==1?12.f:38.f));
            float az=6.28318530718f*((j%4)+(j/4)*.31f)/4.f;
            q.x=s.tx.x+radius*cosf(az);q.y=s.tx.y+radius*sinf(az);
        }else{
            int j=e-36;float t=j/2==0?.3f:.7f;float lateral=j%2==0?-.095f:.095f;
            float dx=x-s.tx.x,dy=y-s.tx.y;
            q.x=s.tx.x+t*dx-lateral*dy;q.y=s.tx.y+t*dy+lateral*dx;
        }
        if(q.x>=0 && q.y>=0 && q.x<=255 && q.y<=255){
            target=sample_ground73(s,q.x,q.y,q.z);q.z+=.035f;
            V delta=physical73(q-p,spacing);dst=sqrtf(dot3(delta,delta));
            dir=unit(delta);Hit block;
            if(first_hit(s,p,dir,dst-.05f,false,block))target=-1;
        }
    }
    if(target<0 || dst<1.f){
        indices[id]=-1;attributes[id]=make_float4(0,0,0,0);return;
    }
    int col=min(63,max(0,int(floorf(((q.x-1.f/3.f)/scale+.5f)/4.f))));
    int row=min(63,max(0,int(floorf(((q.y-1.f/3.f)/scale+.5f)/4.f))));
    V n=flat_normal(s,target),receiver=flat_normal(s,own);
    V incident=unit(physical73(q-s.tx,spacing));
    V spec=incident-n*(2*dot3(incident,n));float departure=-dot3(dir,n);
    indices[id]=row*64+col;
    attributes[id]=make_float4(log1pf(dst)/6.f,departure,dot3(dir,receiver),-dot3(spec,dir));
}
extern "C" int visibility_graph73(const float4 *geometry,const float3 *tx,const float *top,
                                   int batch,int *neighbors,float4 *attributes,cudaStream_t stream) {
    int n=batch*2*nodes73*edges73;
    edges_kernel73<<<(n+255)/256,256,0,stream>>>(geometry,tx,top,batch,neighbors,attributes);
    return cudaGetLastError();
}
