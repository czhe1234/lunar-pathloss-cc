"""Read-only source-data discovery. Public organizer sources; no execution of fetched code."""
from pathlib import Path
import json,hashlib,urllib.request,concurrent.futures,datetime,re,os
R=Path(__file__).resolve().parents[1];O=R/'audit/generation_alignment_20261101';S=O/'official_sources';S.mkdir(exist_ok=True)
items=[
('terrain_generator.py','https://raw.githubusercontent.com/anderspearson206/LunarTerrainGenerator/main/lunar_terrain_generator/lunar_terrain_generator.py','e732920770fd984889cb96061b345d563ccd1024'),
('terrain_example.ipynb','https://raw.githubusercontent.com/anderspearson206/LunarTerrainGenerator/main/example.ipynb','f156485f09e91e7597f00a6ef4012abbc95d721d'),
('competition_dataset.html','https://raw.githubusercontent.com/lunarradiomapchallenge/lunarradiomapchallenge.github.io/main/dataset.html','15c286ee5cdb9c4d18c0c75857e084ae92dda7bc'),
('research_index.html','https://raw.githubusercontent.com/radiolunadiff/radiolunadiff.github.io/main/index.html','af4b43de7839758c17b3552cb129e5f77ef4d362'),
('sample_650_gt_415.html','https://raw.githubusercontent.com/radiolunadiff/radiolunadiff.github.io/main/static/3dterrain/sample_650_3d_terrain_ground_truth_415.html','c11985b73c468ac505a45f6044e407afd1d0ea15'),
('sample_650_gt_58.html','https://raw.githubusercontent.com/radiolunadiff/radiolunadiff.github.io/main/static/3dterrain/sample_650_3d_terrain_ground_truth_5_8.html','1cf0c7142276aafeb6d9b704c2d645235883ad09'),
('sample_680_gt_415.html','https://raw.githubusercontent.com/radiolunadiff/radiolunadiff.github.io/main/static/3dterrain/sample_680_3d_terrain_ground_truth_415.html','55625996c2e9268a0e02d587c33c05f81fe98be2'),
('sample_680_gt_58.html','https://raw.githubusercontent.com/radiolunadiff/radiolunadiff.github.io/main/static/3dterrain/sample_680_3d_terrain_ground_truth_5_8.html','511614573f94907bef14794f17e14e497efffc28')]
def fetch(item):
 name,url,ref=item;p=S/name
 if p.exists():data=p.read_bytes()
 else:
  req=urllib.request.Request(url,headers={'User-Agent':'CC-source-data-audit'})
  with urllib.request.urlopen(req,timeout=45) as f:data=f.read()
  p.write_bytes(data)
 git=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest();assert git==ref,(name,git,ref)
 return {'file':str(p.relative_to(R)),'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha1':git,'expected_blob_matches':True,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 records=list(pool.map(fetch,items))
(O/'official_source_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
for r in records:print(r['file'],r['bytes'],r['expected_blob_matches'])
