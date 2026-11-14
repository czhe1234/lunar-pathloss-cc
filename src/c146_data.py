"""C146: H/TX inputs only; no teacher/model/prediction reads in this module."""
from pathlib import Path
import csv, hashlib, json, math
import numpy as np
import torch
from src.c3_geometry import basic_inputs
from src.c123_lossless import Field

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data"
AUDIT = ROOT / "audit/c146_solver_ablation"
RUN = ROOT / "runs/c146_solver_ablation"
ARMS = ("with_solver", "without_solver")
def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temp.replace(path)
def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()
def digest(net):
    h = hashlib.sha256()
    for k,v in net.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def create(seed):
    torch.manual_seed(seed)
    return Field("native")
def model_inputs(basic, physical, arm):
    assert arm in ARMS and basic.shape[1] == 11 and physical.shape[1] == 45
    # Mask BEFORE both local and TX-FiLM paths. Former C82 slots are ALWAYS zero.
    supplied = physical if arm == "with_solver" else torch.zeros_like(physical)
    return torch.cat((basic, supplied, torch.zeros_like(basic[:, :2])), 1).contiguous(memory_format=torch.channels_last)

class Reader:
    def __init__(self, supervision=True):
        self.rows = list(csv.DictReader((DATA/"train_index.csv").open()))
        self.h = np.load(DATA/"train_hm.npy", mmap_mode="r", allow_pickle=False)
        self.lookup = {}
        for folder in ("c73_compact", "c73_dev_extra_compact"):
            for p in sorted((ROOT/"features"/folder).glob("part*")):
                if not (p/"complete.json").exists(): continue
                a = np.load(p/"features.npy", mmap_mode="r", allow_pickle=False)
                assert a.dtype == np.float16 and a.shape[1:] == (45,256,256)
                for j,i in enumerate(json.loads((p/"indices.json").read_text())):
                    if int(i) in self.lookup: raise ValueError("Duplicate solver feature map")
                    self.lookup[int(i)] = (a,j)
        self.y = self.w = None
        if supervision:
            self.y = [np.load(DATA/f"train_pl{b}.npy", mmap_mode="r", allow_pickle=False) for b in ("415","58")]
            self.w = [np.load(DATA/f"train_mask{b}.npy", mmap_mode="r", allow_pickle=False) for b in ("415","58")]
    def fields(self, ids):
        return np.stack([self.lookup[int(i)][0][self.lookup[int(i)][1]] for i in ids])
    def inputs(self, ids, device):
        rows = [self.rows[int(i)] for i in ids]
        h = torch.as_tensor(np.array(self.h[[int(r["hm_row"]) for r in rows]]),device=device).float()
        tr = torch.tensor([int(r["tx_row"]) for r in rows],device=device)
        tc = torch.tensor([int(r["tx_col"]) for r in rows],device=device)
        basic, fspl = basic_inputs(h,tr,tc)
        physical = torch.as_tensor(self.fields(ids),device=device).float()
        return basic, physical, fspl

class Panel:
    def __init__(self, reader, ids, device="cuda"):
        self.ids = list(ids); self.device = device; n = len(ids)
        self.terrain = [reader.rows[i]["terrain_id"] for i in ids]
        rows = [reader.rows[i] for i in ids]
        unique, inv = np.unique([int(r["hm_row"]) for r in rows],return_inverse=True)
        self.heights = torch.as_tensor(np.array(reader.h[unique]),device=device).float()
        self.hidx = torch.tensor(inv,device=device)
        self.tr = torch.tensor([int(r["tx_row"]) for r in rows],device=device)
        self.tc = torch.tensor([int(r["tx_col"]) for r in rows],device=device)
        self.physical = torch.empty((n,45,256,256),dtype=torch.float16,device=device)
        self.y = torch.empty((n,2,256,256),dtype=torch.float32,device=device)
        self.w = torch.empty((n,2,256,256),dtype=torch.bool,device=device)
        order = sorted(range(n),key=lambda j:(str(reader.lookup[ids[j]][0].filename),reader.lookup[ids[j]][1]))
        for start in range(0,n,32):
            dst=order[start:start+32]; ix=[ids[j] for j in dst]
            idx=torch.tensor(dst,device=device)
            self.physical.index_copy_(0,idx,torch.as_tensor(reader.fields(ix),device=device))
        for start in range(0,n,64):
            ix=ids[start:start+64]
            self.y[start:start+len(ix)] = torch.as_tensor(np.stack([a[ix] for a in reader.y],1),device=device).float()
            self.w[start:start+len(ix)] = torch.as_tensor(np.stack([np.unpackbits(a[ix],axis=1).reshape(-1,256,256) for a in reader.w],1),device=device).bool()
        print(json.dumps({"event":"panel_loaded","maps":n,"terrains":len(set(self.terrain))}),flush=True)
    def batch(self, ix):
        basic, fspl=basic_inputs(self.heights[self.hidx[ix]],self.tr[ix],self.tc[ix])
        return basic,self.physical[ix].float(),fspl,self.y[ix],self.w[ix]
