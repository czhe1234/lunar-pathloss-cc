"""Assemble the reproducible C82 predictor and official-template CSV after scoring."""
import csv,hashlib,json,shutil,sys,tarfile
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
from c76_deliver import export
run=R/'runs/c82_selected';out=R/'releases/C82_20261103'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    off=json.load(open(R/'audit/c82_selected_official/verified_summary.json'));parent=json.load(open(R/'audit/c76_selected_official/verified_summary.json'))
    delta=np.array(parent['RMSE'])-off['RMSE']
    frozen=json.load(open(run/'freeze.json'));assert not frozen['official_VAL_used_for_selection']
    prev=R/'releases/C76_20261103'
    out.mkdir(exist_ok=True);(out/'src').mkdir(exist_ok=True);(out/'src/__init__.py').touch()
    for sub,srcdir,outdir in [('files',run,out),('implementation',R/'src',out/'src')]:
        for name,h in frozen[sub].items():
            assert sha(srcdir/name)==h
            dest=outdir/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(srcdir/name,dest)
    shutil.copy2(run/'freeze.json',out/'freeze.json')
    shutil.copy2(R/'audit/c82_selected_official/verified_summary.json',out/'official_validation.json')
    for name in ('requirements.lock.txt','build_cuda.sh'):shutil.copy2(prev/name,out/name)
    txt=(prev/'infer.py').read_text();assert 'from src.c74_inference' in txt
    (out/'infer.py').write_text(txt.replace('from src.c74_inference','from src.c82_inference'))
    st=[]
    for p in sorted((run/'predictions').glob('*/inference.json')):
        k=json.load(open(p));arr=np.load(p.parent/'seconds_per_map.npy')[1:]
        st.append(dict(hardware=k['gpu'],warm_times=len(arr),mean_s=float(arr.mean()),median_s=float(np.median(arr)),p95_s=float(np.quantile(arr,.95)),max_s=float(arr.max()),over_500_ms=int((arr>.5).sum())))
    (out/'README.md').write_text(f"""# C82 月球双频路损预测

完整官方固定验证的双频 RMSE 为 {off['RMSE'][0]:.6f} / {off['RMSE'][1]:.6f} dB，综合 {off['combined_rmse_db_masked']:.6f} dB。相对 C76 改善 {delta[0]:.6f} / {delta[1]:.6f} dB。两个频段都降低 0.30 dB 才达到大幅提分标准，最终目标是各自不高于 1 dB。

输入为一个 npz 文件，含 256×256 的 float32 高程 heightmap 和整数 tx_row、tx_col。输出为形状 2×256×256 的 npy 文件，依次为 415 MHz 和 5.775 GHz。已包括权重、源文件、已编译的 CUDA 库和编译脚本。验证集标签只在冻结后用于计分。

在 transformer 的 CC 目录，使用 ./run_gpu.sh 3 releases/C82_20261103/infer.py --input 输入.npz --output 预测.npy。完整计时和超过 500 ms 的次数见 timing.json。
""")
    (out/'timing.json').write_text(json.dumps(st,indent=2)+'\n')
    part=R/'releases/C82_20261103.tar.gz'
    with tarfile.open(part.with_suffix('.tmp'),'w:gz',compresslevel=5) as f:f.add(out,arcname=out.name,filter=lambda x: None if '__pycache__' in x.name else x)
    part.with_suffix('.tmp').replace(part)
    export(run,'C82',R/'submissions/C82_Kaggle_20261103')
    delta_json=dict(version='C82',official=off,baseline=parent['RMSE'],delta=delta.tolist(),combined_delta=parent['combined_rmse_db_masked']-off['combined_rmse_db_masked'],
       both_improved=bool((delta>0).all()),both_major=bool((delta>=.30).all()),target=off['goal_both_le_1db'],timing=st,archive_sha256=sha(part),website_submission=False)
    (R/'audit/c79_receiver/official_delivery.json').write_text(json.dumps(delta_json,indent=2)+'\n');print(off['RMSE'],delta.tolist(),sha(part),flush=True)
if __name__=='__main__':main()
