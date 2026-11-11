"""Use the pinned original MeshRadioMap for receiver XY centers, including its floating point sum order."""
import json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from c79_sionna import mi,dr,rt
from sionna.rt.radio_map_solvers.mesh_radio_map import MeshRadioMap
A=R/'audit/c94_measurement'
scene=rt.load_scene();scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V')
scene.add(rt.Transmitter('tx',position=[0,0,3]))
mesh=mi.load_dict({'type':'ply','filename':str(R/'audit/c93_joint_miss/native0/rx.ply'),'face_normals':True})
m=MeshRadioMap(scene,mesh);c=m.cell_centers.numpy().T[:,:2]
np.save(A/'sionna_centers.npy',c)
print(c.shape,c.dtype,c[:2].tolist(),flush=True)
