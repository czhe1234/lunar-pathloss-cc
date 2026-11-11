"""Independent aggregation checks of saved per-map statistics and release arrays."""
from pathlib import Path
import csv,json,math
import numpy as np
root=Path(__file__).resolve().parents[1];out=root/'audit';data=root.parent/'data'
r=json.loads((out/'raw_audit.json').read_text());rows=list(csv.DictReader((out/'map_stats.csv').open()));checks={}
checks['no_reported_data_errors']=not r['issues']
checks['terrain_and_sample_splits_disjoint']=not r['split_checks']['overlap_terrain_ids'] and not r['split_checks']['overlap_sample_ids']
checks['heightmaps_across_splits_not_exact_duplicates']=not r['split_checks']['exact_duplicate_heightmaps_across_splits']
checks['frozen_validation_list_matches']=r['split_checks']['val_terrains_txt_matches_index']
checks['files_stable_during_hash']=all(x['unchanged_during_hash'] for x in r['files'].values())
checks['stats_cover_all_maps']=len(rows)==2*sum(x['samples'] for x in r['splits'].values())
for s,sp in r['splits'].items():
 checks[f'{s}_index_valid']=not sp['duplicate_sample_ids'] and not sp['invalid_index_rows'] and not sp['hm_row_to_multiple_terrains'] and not sp['terrain_to_multiple_hm_rows']
 h=np.load(data/f'{s}_hm.npy',mmap_mode='r',allow_pickle=False)
 checks[f'{s}_heightmap_header']=h.shape==(sp['terrains'],256,256) and h.dtype==np.float32
 checks[f'{s}_heightmap_finite']=sp['heightmaps']['nonfinite_pixels']==0
 for b,st in sp['bands'].items():
  key=f'{s}_{b}'
  a=np.load(data/f'{s}_pl{b}.npy',mmap_mode='r',allow_pickle=False);m=np.load(data/f'{s}_mask{b}.npy',mmap_mode='r',allow_pickle=False)
  checks[key+'_headers']=a.shape==(sp['samples'],256,256) and a.dtype==np.float16 and m.shape==(sp['samples'],8192) and m.dtype==np.uint8
  ss=[v for v in rows if v['split']==s and v['band']==b]
  n=sum(int(v['valid_count']) for v in ss)
  weighted_mean=math.fsum(float(v['valid_mean_db'])*int(v['valid_count']) for v in ss)/n
  checks[key+'_counts']=n==st['valid_pixels']==st['valid_pixel_distribution']['count']
  checks[key+'_mean_crosscheck']=abs(weighted_mean-st['valid_pixel_distribution']['mean'])<1e-8
  checks[key+'_all_values_finite_positive']=st['nonfinite_pixels']==0 and st['nonpositive_pixels']==0
  if b=='58':checks[key+'_fill_matches_mask']=st['invalid_not145_pixels']==0
checks['all_submission_ids_valid']=r['submission_check']['rows']==800000 and r['submission_check']['duplicates']==0 and r['submission_check']['unmatched_ids']==0 and r['submission_check']['rows_on_mask0']==0
alignment=json.loads((out/'input_alignment.json').read_text())
checks['mask_unpack_verified_independently']=all(x['mask_bit_checks_pass'] for x in alignment.values())
result={'checks':{k:bool(v) for k,v in checks.items()},'all_passed':all(checks.values()),'limitations':['Local SHA256 fingerprints have not been compared against an authenticated official checksum list.','Exact duplicate checks do not establish independence of larger parent terrains.','No training or model comparison was performed.']}
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
assert result['all_passed'], 'Audit verification failed'
