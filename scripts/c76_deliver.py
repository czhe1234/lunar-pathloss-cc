"""Build a self-contained server release and verify every public-template CSV prediction."""
from pathlib import Path
import csv,hashlib,itertools,json,os,re,shutil
import numpy as np
R=Path(__file__).resolve().parents[1];DATA=R.parent/'data';name='C76'
run=R/'runs/c76_selected';out=R/'releases/C76_20261103'
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def prepare():
    off=json.load(open(R/'audit/c76_selected_official/verified_summary.json'));base=json.load(open(R/'audit/c65_combo_full_official/verified_summary.json'))
    if not all(x<y for x,y in zip(off['RMSE'],base['RMSE'])):raise ValueError('No joint official-validation improvement; do not replace C65')
    fr=json.load(open(run/'freeze.json'))
    out.mkdir(exist_ok=False);(out/'src').mkdir();(out/'src/__init__.py').touch()
    for rel,digest in fr['files'].items():
        p=run/rel;assert sha(p)==digest
        q=out/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    for f,digest in fr['implementation'].items():
        assert sha(R/'src'/f)==digest
        shutil.copy2(R/'src'/f,out/'src'/f)
    shutil.copy2(run/'freeze.json',out/'freeze.json');shutil.copy2(R/'requirements.lock.txt',out/'requirements.lock.txt')
    shutil.copy2(R/'audit/c76_selected_official/verified_summary.json',out/'official_validation.json')
    (out/'infer.py').write_text("""import argparse,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from src.c74_inference import Predictor
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
d=np.load(a.input,allow_pickle=False);net=Predictor(root)
np.save(a.output,net.predict(d['heightmap'],int(d['tx_row']),int(d['tx_col'])),allow_pickle=False)
""")
    build='''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/src"
compile_cuda() {
    /usr/local/cuda-11.8/bin/nvcc -O3 -Xcompiler -fPIC -shared -gencode arch=compute_80,code=compute_80 -gencode arch=compute_80,code=sm_80 "$@"
}
g++ -O3 -fPIC -fopenmp -shared c66_lattice.cpp -o libc66_lattice.so
compile_cuda c67_transport.cu -o libc67_transport.so
compile_cuda c70_geometry.cu -o libc70_geometry.so
compile_cuda c73_visibility.cu -o libc73_visibility.so
compile_cuda --fmad=false c74_surface.cu -o libc74_surface.so
'''
    (out/'build_cuda.sh').write_text(build);(out/'build_cuda.sh').chmod(0o755)
    info=dict(frequency_order=['415 MHz','5.775 GHz'],official_RMSE=off['RMSE'],official_combined=off['combined_rmse_db_masked'],terrain_count=680,maps_per_band=2330,
        old_C65_RMSE=base['RMSE'],gain=[b-x for b,x in zip(base['RMSE'],off['RMSE'])],input=['heightmap (float32, 256x256, meters)','tx_row, tx_col (integer raster coordinates)'],
        training_labels='official TRAIN only',official_val='used for final scoring and version comparison',hidden_test_or_site_score=False)
    write(out/'interface.json',info)
    (out/'README.md').write_text(f"""# C76 月球双频路损预测

官方固定验证覆盖 680 块地形，每频段 2,330 张图。415 MHz 为 {off['RMSE'][0]:.6f} dB，5.775 GHz 为 {off['RMSE'][1]:.6f} dB。双频综合 RMSE 为 {off['combined_rmse_db_masked']:.6f} dB。

模型选择与权重清单见 freeze.json。正式输出顺序是 415 MHz、5.775 GHz。输入只需 256×256 米制高度图和整数发射机行列坐标。验证标签和掩码仅供外部评价。

在 transformer 的 CC 目录中运行以下命令。输入 NPZ 含 heightmap、tx_row、tx_col。输出 NPY 是两个 256×256 的 dB 路损图。

    ./run_gpu.sh 3 releases/C76_20261103/infer.py --input input.npz --output prediction.npy

包内含用于已授权 A800 和 H100 的二进制与源码。首次运行包括模型初始化与 PyTorch 编译。build_cuda.sh 可在对应 Linux CUDA 环境中重编译。正式平台部署时还需按平台配置设备。

评分使用官方原始脚本与独立 float64 复算。官方固定验证已用于本次版本对比。隐藏测试和网站分数另行产生。
""")
def export(run_dir=run, version=name, dest=None):
    freeze=sha(run_dir/'freeze.json');given=dest is not None
    dest=Path(dest) if given else R/'submissions'/f'{version}_Kaggle_20261103'
    dest.mkdir(exist_ok=given)
    rows=list(csv.DictReader((DATA/'val_index.csv').open()))
    subset={r['sample_id'] for r in csv.DictReader((DATA/'subset_index.csv').open())}
    arrays={}
    for d in sorted((run_dir/'predictions').glob('val_full_c34_fresh_shard*')):
        m=json.load(open(d/'inference.json'));ii=json.load(open(d/'indices.json'))
        assert m['complete'] and m['target_or_masks_used'] is False and m['freeze_sha256']==freeze
        for b in ('415','58'):
            a=np.load(d/f'pl{b}.npy',mmap_mode='r')
            for j,i in enumerate(ii['index_rows']):
                sid=rows[i]['sample_id'];assert sid==ii['sample_ids'][j]
                if sid in subset:arrays[(b,sid)]=(a,j)
    assert len(arrays)==2*len(subset)==200
    pat=re.compile(r'pl(415|58)_([0-9]+_[0-9]+)_([0-9]+)')
    target=dest/f'submission_{version}.csv';partial=target.with_suffix('.csv.part');seen={};count=0
    def value(key):
        m=pat.fullmatch(key);assert m is not None
        band,sample,pixel=m.groups();x=int(pixel);assert 0<=x<65536
        ar,j=arrays[band,sample];return (band,sample,x),ar[j,x//256,x%256]
    with (DATA/'sample_submission.csv').open(newline='') as f,partial.open('w',newline='') as o:
        reader=csv.DictReader(f);assert reader.fieldnames==['ID','PL'];writer=csv.writer(o,lineterminator='\n');writer.writerow(reader.fieldnames)
        for row in reader:
            (band,sid,pixel),v=value(row['ID']);pixels=seen.setdefault((band,sid),set());assert pixel not in pixels;pixels.add(pixel)
            assert np.isfinite(v);writer.writerow([row['ID'],format(float(v),'.9g')]);count+=1
    assert count==800000 and all(len(p)==4000 for p in seen.values())
    verified=0
    with (DATA/'sample_submission.csv').open(newline='') as ft,partial.open(newline='') as fc:
        for ref,actual in itertools.zip_longest(csv.DictReader(ft),csv.DictReader(fc)):
            assert ref and actual and ref['ID']==actual['ID']
            _,v=value(ref['ID']);assert np.float32(float(actual['PL']))==v
            verified+=1
    assert verified==count
    os.replace(partial,target)
    write(dest/'manifest.json',dict(model=version,file=target.name,rows=count,subset_maps_per_band=len(subset),pixels_per_band_map=4000,
        template_ID_sequence_exact=True,all_output_values_match_float32_inference=True,bytes=target.stat().st_size,sha256=sha(target),
        weights_manifest_sha256=freeze,template_sha256=sha(DATA/'sample_submission.csv'),official_labels_used_for_export=False,website_submission=False))
if __name__=='__main__':prepare();export()
