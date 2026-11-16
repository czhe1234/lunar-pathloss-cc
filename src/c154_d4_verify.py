"""Independent scalar C150 oracle checks of C154's largest FSPL discrepancies."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
from src.c150_reference import registered_profile,SPACING_M
from src.c154_d4_consistency import ACTIONS,label,point,spatial,terrain,tx,fspl,QX,QY,S,affine
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"audit/c154_d4"
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def oracle(h,r,c,rr,cc):
    distance,_height,tz,rz=registered_profile(h,r,c,rr,cc)
    length=math.hypot(distance[-1],rz-tz)
    return 20*math.log10(4*math.pi*length*415e6/299792458)
def main():
    records=json.loads((OUT/"geometry_components.json").read_text())["map_action_records"]
    rows=list(csv.DictReader((ROOT.parent/"data/train_index.csv").open()))
    hm=np.load(ROOT.parent/"data/train_hm.npy",mmap_mode="r")
    checks=[]
    for item in records:
        i=item["map_id"];row=rows[i];h=hm[int(row["hm_row"])]
        r,c=int(row["tx_row"]),int(row["tx_col"])
        g=next(x for x in ACTIONS if label(*x)==item["action"])
        hg=spatial(h,*g);rr,cc=point(r,c,*g)
        old=fspl(QX*S,QY*S,terrain(h,QX,QY)+1,tx(h,r,c))[0]
        new=fspl(QX*S,QY*S,terrain(hg,QX,QY)+1,tx(hg,rr,cc))[0]
        diff=new-spatial(old,*g)
        u,v=np.unravel_index(np.abs(diff).argmax(),diff.shape)
        coordinate_ids=spatial(np.arange(65536).reshape(256,256),*g)
        ou,ov=divmod(int(coordinate_ids[u,v]),256)
        independent=oracle(hg,rr,cc,int(u),int(v))-oracle(h,r,c,ou,ov)
        assert abs(independent-float(diff[u,v]))<1e-9
        assert abs(abs(independent)-item["FSPL_component_error_dB"]["canonical_all"]["max_abs"])<1e-9
        checks.append({"sample_id":row["sample_id"],"action":item["action"],
            "max_difference_new_rx_rc":[int(u),int(v)],"original_rx_rc":[ou,ov],
            "signed_FSPL_difference_dB":independent,
            "independent_CPU_oracle_vs_array_abs_dB":abs(independent-float(diff[u,v]))})
    d={"status":"independent_geometry_verification_passed","cases":len(checks),
        "source_sha256":sha(Path(__file__)),"geometry_result_sha256":sha(OUT/"geometry_components.json"),
        "max_oracle_error_dB":max(x["independent_CPU_oracle_vs_array_abs_dB"] for x in checks),
        "checks":checks,"model_inference":False,"training":False,"CUDA":False}
    (OUT/"geometry_verified.json").write_text(json.dumps(d,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in d.items() if k!="checks"}))
if __name__=="__main__":main()
