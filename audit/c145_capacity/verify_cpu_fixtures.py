"""CPU-only behavioral fixtures for C145 recovery and export cache.

Extract helper functions with AST. Importing the full trainer would load
GPU-bound data modules, so these fixtures deliberately do not import it.
"""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "src/c145_train.py"
REPORT = ROOT / "audit/c145_capacity/cpu_fixture_result.json"
names = {
    "file_hash", "save_checkpoint", "prediction_identity", "prediction_sidecar",
    "cached_prediction_metadata", "record_prediction", "export_checkpoints",
    "reconcile_history", "validate_resume_configuration", "load_resume_checkpoint",
    "plateau_stop_ready", "record_completed_export",
}
tree = ast.parse(TRAINER.read_text(), filename=str(TRAINER))
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert {node.name for node in nodes} == names
namespace = dict(hashlib=hashlib, json=json, np=np, os=os, torch=torch)
def atomic_json(path, value):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)
namespace["atomic_json"] = atomic_json
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAINER), "exec"), namespace)
checks = []
def checked(name):
    checks.append(name)
def raises(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")

history = [
    {"additional_epoch": 4, "additional_step": 1408},
    {"additional_epoch": 8, "additional_step": 2816},
    {"additional_epoch": 12, "additional_step": 4224},
]
assert namespace["reconcile_history"](history, 8, 2816) == history[:2]
raises(lambda: namespace["reconcile_history"]([history[0],history[0]], 8, 2816))
raises(lambda: namespace["reconcile_history"]([history[1],history[0]], 8, 2816))
raises(lambda: namespace["reconcile_history"]([history[0],dict(history[1],additional_step=1)], 8, 2816))
raises(lambda: namespace["reconcile_history"]([history[1]], 8, 2815))
checked("history trims uncommitted evaluations and rejects duplicate, unordered, and excess steps")

plan = json.loads((ROOT / "audit/c145_capacity/protocol.json").read_text())
saved = deepcopy(plan)
for key in ("batch", "parent_checkpoint", "parent_checkpoint_sha256", "panels"):
    changed = deepcopy(saved)
    changed[key] = "changed"
    raises(lambda: namespace["validate_resume_configuration"](changed, plan))
changed = deepcopy(saved)
changed["source_sha256"] = {}
changed["initial_phase_additional_epochs"] += 64
namespace["validate_resume_configuration"](changed, plan)
checked("resume rejects changed batch, parent identity, and ordered panels")
state = {
    "learning_rate_reductions": 2, "last_meaningful_fit_improvement_epoch": 0,
    "last_lr_reduction_epoch": 32,
}
assert not namespace["plateau_stop_ready"](state, 32, plan)
assert not namespace["plateau_stop_ready"](state, 47, plan)
assert namespace["plateau_stop_ready"](state, 48, plan)
checked("plateau cannot stop on reduction or before 16 subsequently trained epochs")

with tempfile.TemporaryDirectory(prefix="cpu-fixture-", dir=REPORT.parent) as temporary:
    run = Path(temporary)
    namespace["RUN"] = run
    def checkpoint(epoch, best_fit, best_monitor):
        return {
            "state": {"value": torch.tensor([float(epoch)])},
            "training_state": {
                "additional_epoch": epoch, "additional_step": epoch*352,
                "best_fit_epoch": best_fit, "best_monitor_epoch": best_monitor,
            },
            "history": [record for record in history if record["additional_epoch"] <= epoch],
        }
    # Simulate a crash after best_fit committed, before best_monitor and last.
    torch.save(checkpoint(7,4,4),run/"last.pt")
    torch.save(checkpoint(8,8,8),run/"best_fit.pt")
    torch.save(checkpoint(4,4,4),run/"best_monitor.pt")
    atomic_json(run/"history.json",history)
    recovered, trimmed, source = namespace["load_resume_checkpoint"]()
    assert source == "best_fit.pt" and recovered["training_state"]["additional_epoch"] == 8
    assert trimmed == history[:2]
    assert json.loads((run/"history.json").read_text()) == history[:2]
    for filename in ("last.pt","best_fit.pt","best_monitor.pt"):
        assert torch.load(run/filename,weights_only=False)["training_state"]["additional_epoch"] == 8
    # The next non-evaluation epoch keeps the earlier evaluated history.
    torch.save(checkpoint(9,8,8),run/"last.pt")
    recovered, trimmed, source = namespace["load_resume_checkpoint"]()
    assert source == "last.pt" and trimmed == history[:2]
    checked("recovery chooses latest complete checkpoint, repairs best-save crash window, restores history")

    class Net:
        def load_state_dict(self, value):
            self.value = float(value["value"][0])
    net = Net()
    def digest(value):
        return hashlib.sha256(str(value.value).encode()).hexdigest()
    namespace["digest"] = digest
    calls = []
    def evaluate(value,panel,batch,label,predictions=None):
        calls.append((value.value,tuple(panel.ids)))
        predictions[:] = value.value
        predictions.flush()
    namespace["evaluate"] = evaluate
    panels = {"fit":SimpleNamespace(ids=[11,12]),"monitor":SimpleNamespace(ids=[99])}
    for filename in ("last.pt","best_fit.pt","best_monitor.pt"):
        torch.save(checkpoint(10,10,10),run/filename)
    namespace["export_checkpoints"](net,panels,8)
    assert len(calls) == 2
    for name in panels:
        first=run/f"selected_{name}_prediction.npy"
        second=run/f"fit_best_{name}_prediction.npy"
        assert first.stat().st_ino == second.stat().st_ino
    checked("matching model and panel cache deduplicates by hardlink after validation")
    namespace["export_checkpoints"](net,panels,8)
    assert len(calls) == 2
    checked("valid cached exports require no forward recomputation")
    # Same shape but new model: selected must regenerate; old fit/last stay valid.
    torch.save(checkpoint(11,10,11),run/"best_monitor.pt")
    namespace["export_checkpoints"](net,panels,8)
    assert len(calls) == 4
    assert np.load(run/"selected_fit_prediction.npy")[0,0,0,0] == 11
    assert np.load(run/"fit_best_fit_prediction.npy")[0,0,0,0] == 10
    checked("same-shaped exports from a stale model are regenerated without corrupting prior hardlinks")
    identity=namespace["prediction_identity"](digest(net),panels["fit"])
    path=run/"last_fit_prediction.npy"
    assert namespace["cached_prediction_metadata"](path,identity) is not None
    bad=deepcopy(identity)
    bad["panel_ids_sha256"]=namespace["prediction_identity"](digest(net),SimpleNamespace(ids=[12,11]))["panel_ids_sha256"]
    assert namespace["cached_prediction_metadata"](path,bad) is None
    bad=deepcopy(identity);bad["forward_precision"]="float32"
    assert namespace["cached_prediction_metadata"](path,bad) is None
    bad=deepcopy(identity);bad["shape"]=[1,2,256,256]
    assert namespace["cached_prediction_metadata"](path,bad) is None
    # Alter an array while retaining shape and file size: content hash must fail.
    array=np.load(path,mmap_mode="r+")
    array[0,0,0,0]=123
    array.flush();del array
    assert namespace["cached_prediction_metadata"](path,identity) is None
    namespace["export_checkpoints"](net,panels,8)
    assert len(calls) == 5
    assert np.load(path)[0,0,0,0] == 10
    # Deleting just a sidecar must not cause an unchecked reuse.
    namespace["prediction_sidecar"](run/"selected_monitor_prediction.npy").unlink()
    namespace["export_checkpoints"](net,panels,8)
    assert len(calls) == 6
    checked("panel ordering, precision, shape, content hash, and completion sidecar govern cache validity")

    done={"additional_epoch":48,"stop_reason":"fixture"}
    namespace["record_completed_export"](done,1.0)
    assert json.loads((run/"training_result.json").read_text())["status"] == "training_complete"
    assert json.loads((run/"result.json").read_text())["status"] == "complete_pending_independent_scoring"
    checked("completed export recovers result records after checkpoint-only stop commit")
assert not torch.cuda.is_initialized()
result = {
    "status":"passed", "GPU_initialized":False, "CUDA_VISIBLE_DEVICES":os.environ["CUDA_VISIBLE_DEVICES"],
    "source_sha256":hashlib.sha256(TRAINER.read_bytes()).hexdigest(),
    "fixture_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "checks":checks,
}
atomic_json(REPORT,result)
print(json.dumps(result,indent=2))
