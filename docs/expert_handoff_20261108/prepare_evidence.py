from pathlib import Path
import json,math,hashlib
R=Path('/home/chenzhe/work/lunar-pathloss-rm-challenge/CC')
D=json.loads((R/'audit/c87_val/summary.json').read_text())
E={'as_of':'2026-11-08 after C130','official':D['complete']['official'],'denominator_check':D['denominator_check'],'regions':{k:D['complete']['groups'][k] for k in ['clear','partial','blocked']},'budgets':[]}
for b,g in zip(['415 MHz','5.775 GHz'],D['complete']['groups']['all']):
 v=g['rmse']**2
 row={'band':b,'baseline':g['rmse'],'required_total_sse_reduction_for_1db':1-1/v,'thresholds':{}}
 for thr in [3,5,10]:
  t=g['abs_ge'+str(thr)];p=t['pixel_fraction_in_group'];q=t['fraction_of_total_SSE']
  row['thresholds'][str(thr)]={'n':t['n'],'valid_fraction':p,'sse_fraction':q,'subset_rmse':math.sqrt(v*q/p),'perfect_repair_global_rmse':math.sqrt(v*(1-q)),'required_subset_rmse_for_global_1db_if_others_unchanged':math.sqrt((1-v*(1-q))/p) if 1>v*(1-q) else None}
 row['tail5_sse_reduction_scenarios']={str(a):math.sqrt(v*(1-a*row['thresholds']['5']['sse_fraction'])) for a in [.5,.8,1]}
 E['budgets'].append(row)
sources=['audit/c87_val/summary.json','audit/c79_receiver/official_delivery.json','runs/c82_selected/freeze.json','audit/c129_fitgap/summary.json','audit/c130_projective/summary.json','audit/c130_projective/probe_verified.json','audit/c130_projective/branch_audit.json','audit/c126_statefield/score.json','audit/c125_tail/diagnosis.json','audit/c125_tail/frozen_gate_probe.json','audit/c124_spectrum/score.json','audit/c127_joint/score.json','results/C121_pair_9435.png']
E['sources']={n:{'bytes':(R/n).stat().st_size,'sha256':hashlib.sha256((R/n).read_bytes()).hexdigest()} for n in sources}
E['calculation_note']='Derived only from saved official C82 descriptive statistics. No model, label fitting, new selection, GPU work or training. Threshold scenarios use true residual membership and hypothetical repair; not deployable predictions.'
out=R/'docs/expert_handoff_20261108/evidence.json';out.write_text(json.dumps(E,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'saved':str(out),'budgets':E['budgets']},ensure_ascii=False,indent=2))
