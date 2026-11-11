"""Recover numerical ground-truth illustration data without executing HTML/JS."""
from pathlib import Path
import re,base64,zipfile,io,json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'audit/generation_alignment_20261101';S=O/'official_sources';V=O/'public_examples';V.mkdir(exist_ok=True);summary=[]
types={'Float64Array':'<f8','Float32Array':'<f4','Uint32Array':'<u4','Int32Array':'<i4','Uint8Array':'u1'}
for p in sorted(S.glob('sample_*_gt_*.html')):
 s=p.read_text();matches=re.findall(r'base64Str\s*=\s*["\']([A-Za-z0-9+/=]+)["\']',s);assert len(matches)==1
 with zipfile.ZipFile(io.BytesIO(base64.b64decode(matches[0]))) as z:
  assert z.namelist()==['index.json'];d=json.loads(z.read('index.json'))
 arrays={k:np.frombuffer(base64.b64decode(a['content']),dtype=types[a['type']]).copy() for k,a in d['hashes'].items()};large_points=[a.reshape(-1,3) for k,a in arrays.items() if d['hashes'][k]['vtk']=='vtkPoints' and len(a)==196608];assert len(large_points)==1;pts=large_points[0];scalar=[a for k,a in arrays.items() if d['hashes'][k]['vtk']=='vtkDataArray' and len(a)==65536];assert len(scalar)==1;values=scalar[0]
 x=pts[:,0].astype(int);y=pts[:,1].astype(int);assert np.allclose(pts[:,:2],np.stack((x,y),1)) and x.min()==y.min()==0 and x.max()==y.max()==255
 h=np.empty((256,256));pl=np.empty_like(h);h[y,x]=pts[:,2];pl[y,x]=values
 smallpoints=[a.reshape(-1,3) for k,a in arrays.items() if d['hashes'][k]['vtk']=='vtkPoints' and len(a)<300];cells=[a for k,a in arrays.items() if d['hashes'][k]['vtk']=='vtkCellArray' and len(a)>1000];assert len(cells)==1;cell=cells[0];assert len(cell)%5==0 and np.all(cell.reshape(-1,5)[:,0]==4)
 np.savez_compressed(V/(p.stem+'.npz'),height_yx=h,display_scalar_yx=pl,points=pts,quad_connectivity=cell.reshape(-1,5)[:,1:],display_marker_points=smallpoints[0])
 def clean(v):
  if isinstance(v,dict):return {k:clean(a) for k,a in v.items() if k not in ['content']}
  if isinstance(v,list):return [clean(a) for a in v]
  return v
 (V/(p.stem+'_scene_metadata.json')).write_text(json.dumps(clean(d),indent=2)+'\n')
 marker=smallpoints[0];cx=float(marker[:,0].mean());cy=float(marker[:,1].mean());rec={'file':p.name,'points':len(pts),'regular_grid_xy':True,'quads':len(cell)//5,'triangles':0,'height_min_max':list(map(float,[h.min(),h.max()])),'display_scalar_min_max':list(map(float,[pl.min(),pl.max()])),'marker_points':marker.tolist(),'marker_mean_xy':[cx,cy],'height_at_marker_xy':float(h[round(cy),round(cx)]),'marker_z_above_terrain_range':[float(marker[:,2].min()-h[round(cy),round(cx)]),float(marker[:,2].max()-h[round(cy),round(cx)])]};summary.append(rec);print(json.dumps(rec),flush=True)
(O/'public_example_structure.json').write_text(json.dumps(summary,indent=2)+'\n')
