"""Export a scored, frozen CC model without optimizer or feature caches."""
import os,sys,json,shutil,hashlib,argparse,tarfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();run=ROOT/'runs'/args.run
score_path=run/'predictions/val_full_bf16/independent_score.json';score=json.loads(score_path.read_text());assert score['source_split']=='val' and score['maps_per_band']==2330 and score['all_verification_checks_passed']
meta=json.loads((score_path.parent/'inference.json').read_text());config=json.loads((run/'config.json').read_text());kind='c4' if config['input_channels']==31 else 'c3';out=ROOT/'deliveries'/args.run;out.mkdir(parents=True,exist_ok=False)
ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False);assert hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()==meta['checkpoint_sha256']
torch.save({'model':ck['model'],'width':config['width'],'architecture':kind},out/'weights.pt');del ck
names=['c3_geometry.py',f'{kind}_model.py']+(['c4_geometry.py'] if kind=='c4' else [])
for name in names:
 f=run/'source'/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==config['source_sha256'][name];shutil.copy2(f,out/name)
shutil.copy2(score_path,out/'official_validation_score.json');shutil.copy2(score_path.parent/'inference.json',out/'inference_benchmark.json');shutil.copy2(run/'config.json',out/'training_config.json')
inference = r'''import os
GPU_UUID="GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0"
if os.environ.get("CUDA_VISIBLE_DEVICES")!=GPU_UUID:
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the authorized physical GPU3 UUID before import.")
import ctypes,uuid,importlib
from pathlib import Path
import torch,numpy as np
class Predictor:
    def __init__(self,directory=None):
        assert torch.cuda.device_count()==1
        driver=ctypes.CDLL("libcuda.so.1");assert driver.cuInit(0)==0
        dev=ctypes.c_int();assert driver.cuDeviceGet(ctypes.byref(dev),0)==0
        raw=(ctypes.c_ubyte*16)();assert driver.cuDeviceGetUuid(ctypes.byref(raw),dev)==0
        assert "GPU-"+str(uuid.UUID(bytes=bytes(raw)))==GPU_UUID
        torch.set_num_threads(8);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        root=Path(directory) if directory else Path(__file__).resolve().parent
        ck=torch.load(root/"weights.pt",map_location="cpu",weights_only=True);kind=ck["architecture"]
        self.inputs=importlib.import_module(kind+"_geometry").make_inputs
        self.model=importlib.import_module(kind+"_model").Model(ck["width"]).to("cuda",memory_format=torch.channels_last)
        self.model.load_state_dict(ck["model"]);self.model.eval()
    @torch.inference_mode()
    def predict_many(self,heightmaps,tx_row_col):
        a=np.asarray(heightmaps,dtype=np.float32);rc=np.asarray(tx_row_col,dtype=np.int64)
        assert a.ndim==3 and a.shape[1:]==(256,256) and rc.shape==(len(a),2)
        assert np.isfinite(a).all() and ((rc>=0)&(rc<256)).all()
        h=torch.from_numpy(np.array(a,copy=True)).cuda();r=torch.tensor(rc[:,0],device="cuda");c=torch.tensor(rc[:,1],device="cuda")
        x,fs=self.inputs(h,r,c)
        with torch.autocast("cuda",dtype=torch.bfloat16):pred=self.model(x)+fs
        return pred.float().cpu().numpy()
    def predict(self,heightmap,tx_row,tx_col):
        return self.predict_many(np.asarray(heightmap)[None],[[tx_row,tx_col]])[0]
'''
(out/'inference.py').write_text(inference)
rm=[score['bands'][b]['rmse_db'] for b in ['415','58']];lat=meta['single_request_warm_ms'];met=all(v<=1 for v in rm)
readme=f"""# 独立月球路径损耗模型 {args.run}

官方固定验证集：2,330 张图/频段。415 MHz RMSE = {rm[0]:.6f} dB；5.8 GHz（实际 5.775 GHz）RMSE = {rm[1]:.6f} dB。
双频各不高于 1 dB：{'达到' if met else '未达到'}。这些是发布验证集的原始掩码 pooled RMSE，不是 Kaggle 格式检查分数。
计分由独立 NumPy 和 PyTorch 从同一份保存预测复核。训练、模型选择未使用官方验证标签。

输入为 float32 高度图 (256,256)，单位米，以及整数 TX 行、列。输出 float32 (2,256,256)，顺序 415 MHz、5.775 GHz，单位正路径损耗 dB。推理不读取标签、掩码或预生成缓存。

在 transformer 上运行前，将 CUDA_VISIBLE_DEVICES 设置为 GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0。使用 CC/.venv；在本目录导入 inference.Predictor，调用 predict(heightmap, tx_row, tx_col)。该设备约束遵守用户当前服务器规则，不会自动改用其他卡。

本机 H100 完整双频单张预热推理：均值 {lat['mean']:.3f} ms，P95 {lat['p95']:.3f} ms，最大 {lat['max']:.3f} ms。首次调用 {meta['single_request_cold_ms']:.3f} ms。包括全部几何计算和数据传输，不含模型加载；官方硬件和预热口径尚未确认。

权重、独立输入几何、模型代码及计分记录都在此目录。完整训练记录和来源快照保留在 CC/runs/{args.run}。
"""
(out/'README.md').write_text(readme)
manifest={'run':args.run,'official_validation_rmse_db':dict(zip(['415','58'],rm)),'goal_met':met,'checkpoint_sha256':meta['checkpoint_sha256'],'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(out.with_suffix('.tar.gz'),'w:gz') as tar:tar.add(out,arcname=out.name)
print(json.dumps({'directory':str(out),'archive':str(out.with_suffix('.tar.gz')),'rmse':rm,'goal_met':met}))
