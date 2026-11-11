import json,hashlib
from pathlib import Path
from src.c109_ordered_ray import create
from src.c105_context import digest
R=Path.cwd();A=R/'audit/c109_architecture';A.mkdir(exist_ok=False)
p=json.loads((R/'audit/c108_diagnosis/protocol.json').read_text())
left=create('ordered');right=create('permuted');assert digest(left)==digest(right)
p.update(question='Does path-locality bias improve newTX and newterrain generalization with identical profile values,parameters,initialweights,andbudget?',
 arms=['ordered','permuted'],steps=2000,eval_steps=[200,500,1000,2000],parameters=sum(x.numel() for x in left.parameters()),
 initial_state_sha256=digest(left),data_scope='exactC108n128 384maps;compareC108at2000updates',
 architecture='C105native+axialtrunk plus32step8channelTX-RXprofiledepthwise1Dconvolution implementedas3Dkernels;mean+maxpathpool;nativequeryreadout.',
 control='permuted usesfixedlabel-independentglobalpermutationofallprofilechannelsincludingt;identicalsamplemultisetandparameters;onlypathadjacencychanged.',
 profile='derivedonlyfromsame58H/TX+C82inputs,existingbasic_inputsTXheightconvention,32midpoints,clearance/Fresnelscales/slope/C82excess/t/range',
 selection='minoriginal128monitorjointmeanincludingzeroC82;freezeallarmsbefore484confirm',
 GPU={'ordered':1,'permuted':3},source_sha256=hashlib.sha256((R/'src/c109_ordered_ray.py').read_bytes()).hexdigest())
(A/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
(A/'PLAN.md').write_text('# C109：路径顺序与架构偏置\n\n依据C108早期轨迹，同地形新TX也失败。保持C108n128的384训练图、58通道输入、初始原生/轴向特征、抽样、优化和前2000更新完全一致。新增同参数的路径读取器，唯一对照为32个剖面点按传播顺序或固定打乱。剖面数值集合保留，打乱也保留位置通道，故不伪称新增原始信息。比较效应只约束该路径局部读取方案，不能由失败排除所有架构。全部H/TX导出，不读标签或掩码作输入；不涉及C95数值偏移分支。只有同确认面板双频各改善至少0.30dB才扩训。\n')
print({'parameters':p['parameters'],'steps':p['steps'],'hash':p['initial_state_sha256']})
