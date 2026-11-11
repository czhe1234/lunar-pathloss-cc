"""Package the frozen jointly improved candidate and export prediction-only CSV."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import csv,json,hashlib,shutil,subprocess,tarfile,time
import numpy as np
R=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
off=json.loads((R/'audit/c65_combo_full_official/verified_summary.json').read_text());old=json.loads((R/'audit/c38_full_refit_official/verified_summary.json').read_text())
newv=np.array(off['RMSE']);oldv=np.array(old['RMSE']);joint=bool((newv<oldv).all());goal=bool((newv<=1).all())
bundle=R/'runs/c65_combo_full';manifest=json.loads((bundle/'manifest.json').read_text())
proof={'model':'C65','official':off,'C38_official':old,'gain_vs_C38_db':(oldv-newv).tolist(),'relative_gain_vs_C38':(1-newv/oldv).tolist(),'jointly_better_than_C38':joint,'goal_reached':goal,'weights_manifest_sha256':sha(bundle/'manifest.json'),'weights_roles':manifest['weights'],'official_VAL_use':{'gradients':False,'update_or_checkpoint_selection':False,'frequency_routing_selection':False,'final_scoring':True,'C65_vs_C38_final_comparison_and_delivery_choice':True,'untouched_hidden_test':False},'website_submission':False}
if not joint:
 proof['package']=None;proof['reason']='Candidate did not improve both bands; C38 upload artifact remains recommended baseline.'
 (R/'audit/c65_final_delivery.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof));raise SystemExit(0)
dest=R/'releases/C65_20261102';dest.mkdir(exist_ok=False);(dest/'model').mkdir();(dest/'src').mkdir()
for p in bundle.glob('*.pt'):shutil.copy2(p,dest/'model'/p.name)
shutil.copy2(bundle/'manifest.json',dest/'model/manifest.json')
names=['c60_inference.py','c54_inference.py','c47_inference.py','c34_model.py','c47_model.py','c48_model.py','c49_model.py','c52_model.py','c3_geometry.py','c4_geometry.py','c7_surface_features.py','c8_tx_context.py','c9_extra_gpu.py','c10_phase_geometry.py']
for name in names:shutil.copy2(R/'src'/name,dest/'src'/name)
core=(R/'src/cc_core.py').read_text();prefix=core[:core.index('def check_source_unchanged')];atom=core[core.index('def atomic_json'):core.index('def d4(')]
(dest/'src/cc_core.py').write_text(prefix+atom);(dest/'src/__init__.py').write_text('')
infer="""from pathlib import Path
import argparse,sys,time,json
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
from src.c60_inference import Predictor
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
data=np.load(a.input,allow_pickle=False);h=np.array(data['heightmap'],np.float32);r=int(data['tx_row']);c=int(data['tx_col'])
net=Predictor(R/'model');t=time.time();pred=net.predict(h,r,c);np.save(a.output,pred,allow_pickle=False)
print(json.dumps({'output':a.output,'shape':list(pred.shape),'dtype':str(pred.dtype),'seconds_including_first_compile':time.time()-t,'input':'heightmap and integerTX only'}))
"""
(dest/'infer.py').write_text(infer);shutil.copy2(R/'requirements.lock.txt',dest/'requirements.lock.txt')
(dest/'official_validation.json').write_text(json.dumps(off,indent=2)+'\n')
readme=f"""# C65 服务器复现包

官方固定VAL：415MHz {newv[0]:.10f}dB；5.775GHz {newv[1]:.10f}dB；综合{newv.mean():.10f}dB。
相比C38改善{oldv[0]-newv[0]:.6f}/{oldv[1]-newv[1]:.6f}dB。
目标双频均≤1dB尚未达到；这是小幅改进版本。

模型：冻结全TRAIN C38周期数值嵌入/8成员BatchEnsemble MLP基底；
低频用有序地形剖面注意力残差，高频用定向地表功率输运注意力残差。
按频段固定组合；不读取标签、有效掩码、训练特征缓存或数据索引进行预测。
C65两个残差专家都以全部6120块TRAIN地形、25720图再拟合固定17770步。
原内部DEV已作为TRAIN，官方VAL未用于梯度、选步数或路由。
官方VAL只在权重和路由冻结后计分。没有网站提交。

本包用于transformer服务器复现，保留用户1/3/4号GPU完整UUID限制。
它不是隐藏测试集最终平台适配包。
从CC运行：
./run_gpu.sh 3 releases/C65_20261102/infer.py --input input.npz --output prediction.npy

input.npz含heightmap(float32,256x256米)、tx_row、tx_col(从0开始的整数像素)。
输出prediction.npy为[2,256,256]的路径损耗dB，频段顺序415MHz、5.775GHz。
运行依赖见requirements.lock.txt。第一次调用包括编译，不能把首次编译耗时当作预热后延迟。
原始模型文件及SHA256见model/manifest.json。数值、延迟及官方评分证据在CC/audit/c65_full_refit_20261102。
"""
(dest/'README.md').write_text(readme)
# Exercise the actual packaged entry point on a legitimate H/TX-only input.
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));row=rows[0];H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
audit=R/'audit/c65_full_refit_20261102';inp=audit/'package_input_map0.npz';output=audit/'package_prediction_map0.npy'
np.savez(inp,heightmap=np.array(H[int(row['hm_row'])],np.float32),tx_row=int(row['tx_row']),tx_col=int(row['tx_col']))
with (R/'logs/c65_package_interface.log').open('w') as log:
 code=75
 for retry in range(60):
  # Remove CPU-only CUDA settings: run_gpu selects and verifies the approved full UUID.
  code=subprocess.call(['./run_gpu.sh','3',str(dest/'infer.py'),'--input',str(inp),'--output',str(output)],cwd=R,stdout=log,stderr=subprocess.STDOUT)
  if code!=75:break
  time.sleep(1)
assert code==0,code
reference=audit/'interface_H100_map0.npy'
assert reference.exists(),'Missing original public-interface reference; do not claim package verified.'
pr=np.load(output);ref=np.load(reference);delta=pr.astype(float)-ref.astype(float);maximum=float(abs(delta).max());rms=np.sqrt((delta*delta).mean((1,2))).tolist()
assert maximum<.005,(maximum,rms)
qa={'actual_packaged_entry_executed':True,'H_TX_only_input':True,'max_abs_diff_from_frozen_public_interface_db':maximum,'RMS_diff_db':rms,'source_data_or_label_cache_required':False}
(audit/'package_interface.json').write_text(json.dumps(qa,indent=2)+'\n')
archive=R/'releases/C65_20261102.tar.gz'
with tarfile.open(archive,'x:gz') as tar:tar.add(dest,arcname=dest.name)
with tarfile.open(archive,'r:gz') as tar:
 for role,info in manifest['weights'].items():
  actual=hashlib.sha256(tar.extractfile(dest.name+'/model/'+role+'.pt').read()).hexdigest();assert actual==info['sha256']
proof.update({'package':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'archive_weights_verified':True,'package_interface':qa})
(R/'audit/c65_final_delivery.json').write_text(json.dumps(proof,indent=2)+'\n')
with (R/'logs/c65_csv_export.log').open('w') as log:subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/export_c65_kaggle.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
csvmeta=json.loads((R/'submissions/C65_Kaggle_20261102/manifest.json').read_text());proof['CSV']={k:csvmeta[k] for k in ['submission_file','row_count','bytes','sha256','all_values_match_original_inference_float32','ground_truth_or_masks_read_for_export']}
(R/'audit/c65_final_delivery.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
