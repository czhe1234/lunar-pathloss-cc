// Fused evaluation of CC's existing C7 directional surface statistics.
#include <cuda_runtime.h>
#include <cmath>
constexpr int side=256,pixels=65536;
__device__ float interp(const float *field,int i00,int i01,int i10,int i11,double w00,double w01,double w10,double w11){
    return float(w00*field[i00]+w01*field[i01]+w10*field[i10]+w11*field[i11]);
}
__global__ void surface_statistics(const float*fields,const double *sampling,int tr,int tc,float *out) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=pixels)return;
    int row=p/256,col=p%256;float zrx=fields[p]+1.f,ht=fields[tr*256+tc];
    double acc[2][21]={};
    for(int az=0;az<16;++az){
        double horizon=-1e6;
        for(int j=0;j<16;++j){
            const double *sample=sampling+(az*16+j)*6;
            double ox=sample[0],oy=sample[1],rad=sample[2],area=sample[3];
            double qx=col+ox,qy=row+oy;
            if(qx<0||qx>255||qy<0||qy>255)continue;
            // Reproduce grid_sample's align_corners=True coordinate conversion.
            double px=fmin(255.,fmax(0.,((qx/127.5-1.)+1.)*127.5)),py=fmin(255.,fmax(0.,((qy/127.5-1.)+1.)*127.5));
            int ix=int(px),iy=int(py),ix1=min(255,ix+1),iy1=min(255,iy+1);
            int i00=iy*256+ix,i01=iy*256+ix1,i10=iy1*256+ix,i11=iy1*256+ix1;
            double wx=px-ix,wy=py-iy,w00=(1-wx)*(1-wy),w01=wx*(1-wy),w10=(1-wx)*wy,w11=wx*wy;
            float hh=interp(fields,i00,i01,i10,i11,w00,w01,w10,w11);
            float sx=interp(fields+pixels,i00,i01,i10,i11,w00,w01,w10,w11),sy=interp(fields+2*pixels,i00,i01,i10,i11,w00,w01,w10,w11);
            float norm=sqrtf(1.f+sx*sx+sy*sy),rz=zrx-(hh+.05f);
            double rlen=sqrt(rad*rad+double(rz*rz));
            double coso=fmax(0.,(sx*ox+sy*oy+rz)/norm/rlen);
            double slope=(hh-zrx)/rad;
            bool visible=slope+.05/rad>=horizon;horizon=fmax(horizon,slope);
            #pragma unroll
            for(int band=0;band<2;++band){
                double *a=acc[band];a[20]+=1;
                if(!visible)continue;
                float txz=(ht+(3-band))-(hh+.05f),dz=(zrx-ht)-(3-band);
                double tx=tc-qx,ty=tr-qy,tl=fmax(.25,sqrt(tx*tx+ty*ty+double(txz*txz)));
                double ci=fmax(0.,(-sx*tx-sy*ty+txz)/norm/tl);
                double d0=(col-tc)*(col-tc)+(row-tr)*(row-tr)+double(dz*dz);
                double geom=area*norm*ci*coso/(tl*tl*rlen*rlen)*d0/3.14159265358979323846;
                const float *v=fields+(3+3*band)*pixels;
                double vis=interp(v,i00,i01,i10,i11,w00,w01,w10,w11);
                double wt=geom*vis;
                a[0]+=wt;a[j<5?1:(j<10?2:3)]+=wt;
                a[4]+=geom*interp(v+pixels,i00,i01,i10,i11,w00,w01,w10,w11);
                a[5]+=geom*interp(v+2*pixels,i00,i01,i10,i11,w00,w01,w10,w11);
                double extra=fmax(0.,tl+rlen-sqrt(d0));
                double align=fmin(1.,fmax(-1.,(tx*ox+ty*oy-double(txz*rz))/(tl*rlen)+2*ci*coso));
                if(wt>a[6]){a[6]=wt;a[18]=extra;a[19]=rad;}
                a[7]+=wt*extra;a[8]+=wt*rad;a[9]+=wt*tl;a[10]+=wt*ci;a[11]+=wt*coso;a[12]+=wt*align;
                double sharp=fmax(0.,align);
                #pragma unroll
                for(int k=0;k<5;++k)sharp*=sharp;
                a[13]+=wt*sharp;a[14]+=wt*sample[4];a[15]+=wt*sample[5];
                a[16]+=vis;a[17]+=vis*area*norm;
            }
        }
    }
    #pragma unroll
    for(int band=0;band<2;++band){
        float *o=out+p*40+20*band;double *a=acc[band],den=fmax(a[0],1e-20);
        for(int j=0;j<7;++j)o[j]=log10(a[j]+1e-16);
        for(int j=7;j<10;++j)o[j]=log1p(a[j]/den);
        for(int j=10;j<13;++j)o[j]=a[j]/den;
        o[13]=log10(a[13]+1e-16);o[14]=a[14]/den;o[15]=a[15]/den;
        o[16]=a[16]/fmax(a[20],1.);o[17]=log1p(a[17]);o[18]=a[0]>0?log1p(a[18]):0.;o[19]=a[0]>0?log1p(a[19]):0.;
    }
}
extern "C" int surface74(const float *fields,const double *sampling,int tr,int tc,float *out,cudaStream_t stream){
    surface_statistics<<<(pixels+127)/128,128,0,stream>>>(fields,sampling,tr,tc,out);return cudaGetLastError();
}
