"""Re-export and verify frozen C82 predictions without training or GPU use."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import hashlib,json,tarfile,sys,shutil
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
from c76_deliver import export
dest=R/'submissions/C82_提交版_20261110';dest.mkdir(exist_ok=False)
run=R/'runs/c82_selected';frozen=json.loads((run/'freeze.json').read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert frozen['official_VAL_used_for_selection'] is False
for rel,h in frozen['files'].items():assert sha(run/rel)==h,rel
archive=R/'releases/C82_20261103.tar.gz'
saved=json.loads((R/'audit/c79_receiver/official_delivery.json').read_text())
assert sha(archive)==saved['archive_sha256']
with tarfile.open(archive,'r:gz') as t:
 for section,prefix in [('files','C82_20261103/'),('implementation','C82_20261103/src/')]:
  for rel,h in frozen[section].items():
   assert hashlib.sha256(t.extractfile(prefix+rel).read()).hexdigest()==h,rel
export(run,'C82',dest)
m=json.loads((dest/'manifest.json').read_text())
old=json.loads((R/'submissions/C82_Kaggle_20261103/manifest.json').read_text())
assert m['sha256']==old['sha256']
score=R/'audit/c82_selected_official/verified_summary.json';shutil.copy2(score,dest/'official_validation.json')
base_manifest=json.loads((run/'weights/base/manifest.json').read_text())
assert base_manifest['official_VAL_used_for_training_or_selection'] is False
readme="""# C82 Kaggle 提交版

整理并核验日期：2026-11-10。

上传文件：submission_C82.csv。

上传入口：https://www.kaggle.com/t/1e9cf1fa5c774010a6922e98abb60449

提交描述可填：
C82 frozen TRAIN-only model. Official validation masked RMSE: 415 MHz 2.246377 dB; 5.775 GHz 2.579150 dB; 50/50 mean 2.412764 dB.

文件包含 800,000 行预测，列名为 ID、PL。ID 和顺序逐行匹配发布的 sample_submission.csv。
覆盖当前 Kaggle 模板的每频段 100 张图、每图 4,000 个指定像素。
全部 PL 均为有限的正路径损耗 dB，逐行回转 float32 后与 C82 冻结预测一致。

模型使用官方 TRAIN 训练。C82 冻结记录注明，官方 VAL 未参与本版本拟合或选择。
官方 VAL 标签用于冻结后的计分；本次导出没有读取标签，也没有用真值替换预测。

完整官方验证集：每频段 2,330 张图。
415 MHz 有效像素 RMSE：2.2463774548 dB。
5.775 GHz 有效像素 RMSE：2.5791498899 dB。
双频等权平均：2.4127636723 dB。
这是完整官方验证结果。当前 Kaggle 模板是其中的指定子集，网站分数需上传后读取。

该 CSV 与 2026-11-03 整理的 C82 CSV 完全一致。没有新的模型改动。
权重和推理代码另存于 CC/releases/C82_20261103.tar.gz。
这份目录用于当前 Kaggle CSV 上传。比赛最终模型交付需按当时的官方要求另行准备。

本次尚未上传网站。
"""
(dest/'README_上传说明.md').write_text(readme)
check=dict(status='passed',verified_date='2026-11-10',version='C82',
 CSV_rows=m['rows'],CSV_bytes=m['bytes'],CSV_sha256=m['sha256'],
 template_ID_sequence_exact=m['template_ID_sequence_exact'],
 all_800000_values_equal_frozen_float32_predictions=m['all_output_values_match_float32_inference'],
 frozen_weights_and_release_source_hashes_match=True,CSV_identical_to_20261103=True,
 official_VAL_used_for_training=False,official_labels_used_for_export=False,
 GPU_used=False,new_training=False,website_submission=False,
 release_archive=str(archive),release_sha256=sha(archive))
(dest/'verification_20261110.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(directory=str(dest),verification=check),ensure_ascii=False),flush=True)
