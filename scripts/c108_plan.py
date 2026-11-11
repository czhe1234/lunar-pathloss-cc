"""Nested terrain-count experiment with a held-out TX on every terrain."""
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from src.c105_context import create,digest
R=Path.cwd();A=R/'audit/c108_diagnosis';A.mkdir(exist_ok=True)
rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')))
cache=json.loads((R/'audit/c101_residual/base0of1/indices.json').read_text());by=defaultdict(list)
for i in cache:by[rows[i]['terrain_id']].append(i)
multi=sorted(t for t,v in by.items() if len(v)==4);assert len(multi)==277
rng=np.random.default_rng(108011);order=[multi[i] for i in rng.permutation(len(multi))]
train_terrain=order[:128];unseen=order[128:160];train={};newtx={};common=[]
for t in train_terrain:
 ids=sorted(by[t]);p=rng.permutation(4);train[t]=[ids[j] for j in p[:3]];newtx[t]=ids[p[3]]
for t in train_terrain[:8]:common.append(newtx[t])
sets={str(n):dict(terrains=train_terrain[:n],fit=sum([train[t] for t in train_terrain[:n]],[]),
 same_terrain_newTX=[newtx[t] for t in train_terrain[:n]]) for n in [8,32,128]}
old=json.loads((R/'audit/c105_context/protocol.json').read_text())
p=dict(question='Separate terrain diversity,unseenTXonsamegeometry,andunseengeometrygeneralization of frozenC82 residuals',
 arms=['n8','n32','n128'],sets=sets,same_terrain_newTX_common8=common,unseen_residual_fit_terrain32=sum([sorted(by[t]) for t in unseen],[]),
 original_monitor=old['monitor'],steps=4000,batch=8,eval_steps=[200,500,1000,2000,4000],seed=105041,sample_seed=108073,
 parameters=sum(v.numel() for v in create().parameters()),initial_state_sha256=digest(create()),
 optimizer='AdamWlr4e-4wd.01constant;clip2;maskedbandMSE/25;nolabeldependenttailweights',fixedTX_perterrain=3,
 scope='bounded diagnostic24/96/384trainingmaps;notfullTRAINrefit',sampling='uniformmaps=uniformterrainsbecause3TXeach',
 checkpoint_selection='minjointmeanfulloriginal128terrainmonitorRMSEincludingC82;saveallfivetimepoints;freezeallarmsbeforeconfirmation',
 comparison='fixedupdatesprimary;reportFITnormalizedSSEandtrajectories;differentpermapexposureisexplicitnotconfoundedwithallegedconvergence',
 source='allCCinputcachesandtop-levelrawTRAIN;C99/C82parentshaveusedFITincluding32internalnewgeometries. Original128monitor/484confirm excluded fromparents.',
 promotion='same484terrain1934map2048pointconfirm,bothgain>=.30dBandpairedterrain95lower>0;thenfullnativebeforefullTRAIN',
 input_information='exactH/TXduplicateauditseparate;negativelearningresultsalonecannotproveinputincompleteness',
 GPU={'n8':1,'n32':3,'n128':4},official_val_used=False)
assert not (A/'protocol.json').exists();(A/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
(A/'PLAN.md').write_text('# C108：地形数量与迁移层级\n\n固定原架构、输入、初始状态、每地形3个TX和4000更新，比较8/32/128块不同地形。每块留1个未用于本轮监督的TX；三组共同看最初8块地形的留出TX、另32块地形、原128监测和484确认。主要看等更新轨迹，并记录训练相对C82平方误差下降，不能把更大组较少的重复次数隐藏。\n\n受限诊断总计24/96/384张训练图。没有明显双频收益不全TRAIN扩训。官方VAL不参与。输入不充分另用精确输入重复对照检验，不用训练失败充当信息下限证明。\n\n沿用experiment受控流程；当前没有bash_exec/artifact配套工具，使用CC既有启动器、JSON日志和本地记录。\n')
(A/'CHECKLIST.md').write_text('# C108\n\n- [x] 读取CC规则、资源与旧结果。\n- [x] 固定不看标签的地形/TX嵌套划分。\n- [x] 完成原始H/TX完全相同样本审计。\n- [ ] 同初始化地形数量受限训练。\n- [ ] 分层迁移、训练拟合程度与新地形对照。\n- [ ] 根据证据选择同数据架构对照。\n- [ ] 所有选择冻结后原484地形确认。\n- [ ] 核对、记录结论与资源释放。\n')
print({'maps':{n:len(v['fit']) for n,v in sets.items()},'common_newTX':len(common),'unseen':len(p['unseen_residual_fit_terrain32']),'parameters':p['parameters']})
