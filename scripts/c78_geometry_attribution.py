"""Partition the original C76 clear set without changing its denominator."""
import sys,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c78_clear'
def read_parts(split,kind):
    ds=sorted(A.glob(f'{split}_{kind}_*of*'))
    for p in ds:assert (p/'complete.json').is_file(),p
    cols=json.load(open(ds[0]/'complete.json'))['columns'];total=sum(len(np.load(p/'positions.npy')) for p in ds)
    arr=None;seen=np.zeros(total,bool)
    for p in ds:
        k=np.load(p/'positions.npy');t=np.load(p/('geometry.npy' if kind=='geom' else 'power.npy'),mmap_mode='r')
        if arr is None:arr=np.empty((total,*t.shape[1:]),t.dtype)
        assert not seen[k].any();arr[k]=t;seen[k]=1
    assert seen.all()
    return arr,cols
def report(split):
    panel=np.load(A/f'{split}_panel.npz');ref=np.load(A/f'{split}_reference.npy',mmap_mode='r')
    geo,cols=read_parts(split,'geom')
    assert (ref[...,6]>=0).all()
    native=ref[...,6]>.99
    groups={'all_maps':np.ones(len(ref),bool)}
    if split=='train':groups.update(cal128=panel['calibration'],confirm484=~panel['calibration'])
    thin=geo[...,:11].min(2);fine_thin=np.minimum(geo[...,1],geo[...,2])
    exact=geo[...,13]>.5;medium=geo[...,14]>.5;wide=geo[...,15]>.5;inside=geo[...,21]>.5
    strictclear=(thin>=.99);stable=strictclear&(geo[...,11:13].min(2)>=.99)
    out={'scope':split,'samples_per_map':panel['pixels'].shape[1],'sampled_maps':len(ref),'nominal_bin_disagreement':None}
    if split in ('train','val'):
        out['nominal_bin_disagreement']={'count':int(np.sum(native!=(geo[...,0]>.99))),'total':int(native.size),'max_fraction_delta':float(np.max(np.abs(ref[...,6]-geo[...,0])))}
    for group,inc in groups.items():
        select=inc[:,None];stats={}
        for band,f in enumerate([415e6,5.775e9]):
            valid=(ref[...,4+band]>.5)&select
            base=native&valid;err=ref[...,band]-ref[...,2+band];e2=err*err
            fs_max=20*np.log10((4*np.pi*f/299792458)*geo[...,17].clip(.001))
            fs_min=20*np.log10((4*np.pi*f/299792458)*geo[...,16].clip(.001))
            fs=20*np.log10((4*np.pi*f/299792458)*geo[...,18].clip(.001))
            deficit=ref[...,band]-fs_max
            surplus=fs_min-ref[...,band]
            bins={'original_clear':native,'switch90':thin<.90,'switch50':thin<.50,'stable99':strictclear,'mid90to99':(thin>=.90)&(thin<.99),'stable_coarse':stable,
                  'fine_mesh_switch90':fine_thin<.90,'nominal_certificate':exact,'two_mesh_altitude_certificate':medium,'wide_certificate':wide,'inside_wide':wide&stable,
                  'stable_deficit3':strictclear&(deficit>3),'stable_surplus3':strictclear&(surplus>3),'stable_within3':strictclear&(deficit<=3)&(surplus<=3),
                  'wide_deficit3':wide&(deficit>3),'wide_surplus3':wide&(surplus>3),'wide_within3':wide&(deficit<=3)&(surplus<=3),'all_deficit3':deficit>3,'all_surplus3':surplus>3}
            denom=e2[base].sum(dtype='float64');num=base.sum();big=(abs(err)>=3);huge=(abs(err)>=5)
            bden=e2[base&big].sum(dtype='float64');hden=e2[base&huge].sum(dtype='float64')
            def one(q):
                e=err[q].astype('float64');n=len(e)
                total=float(e@e)
                return dict(n=n,pixel_share=n/num,sse_share=total/denom,rmse=(total/n)**.5 if n else None,bias=float(e.mean()) if n else None,
                    big_sse_share=e2[q&big].sum(dtype='float64')/bden,huge_sse_share=e2[q&huge].sum(dtype='float64')/hden,
                    frac_abs_ge3=float(big[q].mean()) if n else None,frac_abs_ge5=float(huge[q].mean()) if n else None,
                    deficit_over1=float((deficit[q]>1).mean()) if n else None,deficit_over3=float((deficit[q]>3).mean()) if n else None,deficit_over6=float((deficit[q]>6).mean()) if n else None,
                    surplus_over3=float((surplus[q]>3).mean()) if n else None,pred_too_bright_sse=float(e[e>0]@e[e>0])/total if total else None)
            stats[str(band)]={key:one(base&b) for key,b in bins.items()}
            stats[str(band)]['all_valid_RMSE']=float(np.sqrt(e2[valid].mean(dtype='float64')))
            stats[str(band)]['original_clear_vs_all_SSE']=float(denom/e2[valid].sum(dtype='float64'))
        # Use identical jointly valid pixels for frequency-common/difference decomposition.
        joint=native&(ref[...,4]>.5)&(ref[...,5]>.5)&select
        paired={}
        for name,q in [('original',joint),('switch90',joint&(thin<.90)),('stable99',joint&strictclear),('wide_cert',joint&wide)]:
            lo=(ref[...,0]-ref[...,2])[q].astype('float64');hi=(ref[...,1]-ref[...,3])[q].astype('float64')
            c=(hi+lo)/2;d=(hi-lo)/2;ce=c@c;de=d@d
            paired[name]=dict(count=len(lo),common_sse_fraction=ce/(ce+de),difference_sse_fraction=de/(ce+de),correlation=float(np.corrcoef(lo,hi)[0,1]),frequency_difference_RMSE=float(2*np.sqrt(de/len(d))))
        out[group]=dict(bands=stats,paired=paired)
    (A/f'{split}_geometry_attribution.json').write_text(json.dumps(out,indent=2)+'\n')
    target=out.get('confirm484',out['all_maps'])
    for b,d in target['bands'].items():
        print(b,[(k,round(d[k]['pixel_share'],3),round(d[k]['sse_share'],3),round(d[k]['rmse'],3)) for k in ['original_clear','switch90','stable99','nominal_certificate','two_mesh_altitude_certificate','wide_certificate','stable_deficit3','stable_surplus3']])
if __name__=='__main__':report(sys.argv[1])
