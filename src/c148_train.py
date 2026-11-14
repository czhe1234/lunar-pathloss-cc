"""C148: one L2-SP arm continued from C145 selected12; no control rerun."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cc_core import (APPROVED_GPU_UUIDS, GPU_UUID, atomic_json,
                         check_source_unchanged, gpu_guard)
from src.c105_context import ResidualPanel, digest, reader
from src.c123_lossless import create
from src.c145_train import evaluate, save_checkpoint, checkpoint_data, file_hash
from src import cc_tracking
import numpy as np
import torch
from tqdm import tqdm

AUDIT = ROOT / "audit/c148_generalization"
RUN = ROOT / "runs/c148_parameter_anchor"
PARENT_RUN = ROOT / "runs/c145_capacity_native"
PARENT_SWANLAB_ID = "ccb84e96260d83de2477d"
BANDS = ("415MHz", "5p775GHz")
C82 = {"fit": [2.184470152566454, 2.471133742110456],
       "monitor": [2.1463905073492664, 2.41801448492553]}


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(state):
    result = hashlib.sha256()
    for name, value in state.items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def parameter_distance(net, anchor):
    """Differentiable FP32 sum; anchor tensors are independent detached clones."""
    return torch.stack([(parameter - anchor[name]).square().sum()
                        for name, parameter in net.named_parameters()
                        if parameter.requires_grad]).sum()


def read_plan():
    protocol = AUDIT / "protocol.json"
    plan = json.loads(protocol.read_text())
    base_path = ROOT / plan["base_protocol"]
    if file_hash(base_path) != plan["base_protocol_sha256"]:
        raise ValueError("C145 frozen protocol changed")
    base = json.loads(base_path.read_text())
    for filename, expected in plan["source_sha256"].items():
        if file_hash(ROOT / filename) != expected:
            raise ValueError(f"Frozen source changed: {filename}")
    required = {"l2sp_lambda": .1, "anchor_norm_sq": 25448.7329673,
                "additional_epochs": 32, "parent_absolute_epoch": 12,
                "parent_additional_epoch": 4, "parent_optimizer_step": 1408}
    for key, expected in required.items():
        if plan.get(key) != expected:
            raise ValueError(f"C148 frozen setting differs: {key}")
    if plan.get("allow_shared") is not True:
        raise ValueError("Protocol must record the authorized sharing policy")
    if plan.get("gpu_uuid") != GPU_UUID or GPU_UUID not in APPROVED_GPU_UUIDS:
        raise ValueError("Launch UUID differs from frozen protocol/allowed devices")
    parent = ROOT / plan["parent_checkpoint"]
    if file_hash(parent) != plan["parent_checkpoint_sha256"]:
        raise ValueError("C145 selected checkpoint changed")
    checkpoint = torch.load(parent, map_location="cpu", weights_only=False)
    for key in ("state", "optimizer", "scheduler", "data_generator",
                "torch_rng", "cuda_rng", "training_state", "history"):
        if key not in checkpoint:
            raise ValueError(f"Parent is missing full continuation state: {key}")
    if state_hash(checkpoint["state"]) != plan["anchor_model_sha256"]:
        raise ValueError("Anchor model identity differs")
    state = checkpoint["training_state"]
    if (state["additional_epoch"], state["additional_step"]) != (4, 1408):
        raise ValueError("Expected selected12 / C145 epoch4, update1408")
    steps = {int(value["step"].item()) for value in checkpoint["optimizer"]["state"].values()}
    if steps != {1408}:
        raise ValueError(f"Parent optimizer steps differ: {steps}")
    net = create("native")
    net.load_state_dict(checkpoint["state"], strict=True)
    anchor_norm = sum(parameter.double().square().sum().item()
                      for parameter in net.parameters() if parameter.requires_grad)
    if not math.isclose(anchor_norm, plan["anchor_norm_sq"], rel_tol=1e-10):
        raise ValueError("Anchor normalization differs from the frozen FIT-only diagnosis")
    generator = torch.Generator().manual_seed(base["data_seed"])
    for _ in range(plan["parent_absolute_epoch"]):
        torch.randperm(len(base["panels"]["fit"]), generator=generator)
    if not torch.equal(generator.get_state(), checkpoint["data_generator"]):
        raise ValueError("Parent data permutation RNG is not the documented 12-epoch continuation")
    verified = json.loads((PARENT_RUN / "selected_verified.json").read_text())
    if verified["export_identity"]["model_sha256"] != plan["anchor_model_sha256"]:
        raise ValueError("Initial verified metrics do not belong to the anchor")
    initial = deepcopy(checkpoint["history"][-1])
    if initial["absolute_epoch"] != 12:
        raise ValueError("Parent history does not end at selected12")
    for panel in ("fit", "monitor"):
        reference = verified["panels"][panel]["groups"]["all"]["RMSE_dB"]
        if not np.allclose(initial["evaluation"][panel]["rmse"], reference, atol=1e-10, rtol=0):
            raise ValueError(f"Saved initial metrics and independent verifier differ: {panel}")
        sidecar = json.loads((PARENT_RUN / f"selected_{panel}_prediction.json").read_text())
        ids_hash = hashlib.sha256(np.asarray(base["panels"][panel], dtype=np.int64).tobytes()).hexdigest()
        prediction = PARENT_RUN / f"selected_{panel}_prediction.npy"
        if (sidecar["model_sha256"] != plan["anchor_model_sha256"]
                or sidecar["panel_ids_sha256"] != ids_hash
                or prediction.stat().st_size != sidecar["file_size_bytes"]):
            raise ValueError(f"Parent prediction identity differs: {panel}")
    check_source_unchanged()
    report = {"status": "CPU_preflight_passed", "protocol_sha256": file_hash(protocol),
              "trainer_sha256": file_hash(Path(__file__)), "parent_checkpoint_sha256": file_hash(parent),
              "anchor_model_sha256": digest(net), "parameters": sum(p.numel() for p in net.parameters()),
              "anchor_norm_sq_actual": anchor_norm, "optimizer_steps": sorted(steps),
              "parent_absolute_epoch": 12, "parent_additional_epoch": 4,
              "parent_additional_step": 1408,
              "rng_sha256": {key: tensor_hash(checkpoint[key])
                             for key in ("data_generator", "torch_rng", "cuda_rng")},
              "expected_final_absolute_epoch": 44,
              "expected_final_additional_step": 12672,
              "initial_metrics_source": "runs/c145_capacity_native/selected_verified.json",
              "official_VAL_used": False}
    return plan, base, checkpoint, initial, report


def device_snapshot():
    query = ["nvidia-smi", "-i", GPU_UUID,
             "--query-gpu=uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    gpu = subprocess.check_output(query, text=True).strip()
    jobs = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    return {"gpu_query": gpu, "co_resident_processes": [row for row in jobs if GPU_UUID in row],
            "allow_shared": True, "other_processes_terminated": False}


@torch.inference_mode()
def verify_parent_forward(net, panels, batch):
    net.eval()
    result = {}
    for name, panel in panels.items():
        selection = slice(0, batch)
        x, baseline, _, valid = panel.batch(selection)
        with torch.autocast("cuda", torch.bfloat16):
            actual = baseline + net(x).float()
        expected = np.load(PARENT_RUN / f"selected_{name}_prediction.npy", mmap_mode="r")[selection]
        difference = (actual.cpu().numpy().astype(np.float64) - expected.astype(np.float64))
        maximum = float(np.abs(difference).max())
        result[name] = {"maps": len(actual), "max_abs_prediction_difference_dB": maximum,
                        "RMSE_prediction_difference_dB": float(np.sqrt(np.square(difference).mean())),
                        "comparison": "same first ordered batch; saved Float32 BF16 parent prediction"}
        # A strict forward-identity guard, rather than a new full-panel parent evaluation.
        if maximum > 1e-5:
            raise ValueError(f"Restored parent BF16 prediction differs on {name}: {maximum}dB")
    net.train()
    return result


def restore_rng(checkpoint, generator):
    generator.set_state(checkpoint["data_generator"])
    torch.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state(checkpoint["cuda_rng"])
    actual = {"data_generator": generator.get_state(), "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state()}
    for key, value in actual.items():
        if not torch.equal(value, checkpoint[key]):
            raise ValueError(f"RNG restore failed: {key}")
    return {key: tensor_hash(value) for key, value in actual.items()}


def metric_log(record, elapsed, eta, total_epoch=44):
    values = {"progress/epoch": record["absolute_epoch"], "progress/total_epochs": total_epoch,
              "progress/updates": record["additional_step"], "time/elapsed_seconds": elapsed,
              "time/eta_seconds": eta, "optim/learning_rate": record["learning_rate"],
              "optim/proposed_lr": record["learning_rate"]}
    for panel, metrics in record["evaluation"].items():
        for channel, band in enumerate(BANDS):
            values[f"{panel}/RMSE_{band}_dB"] = metrics["rmse"][channel]
            values[f"{panel}/tail_RMSE_{band}_dB"] = metrics["groups"]["tail"]["RMSE"][channel]
            values[f"{panel}/C82_good_RMSE_{band}_dB"] = metrics["groups"]["C82_good"]["RMSE"][channel]
    values["train/data_loss"] = record["evaluation"]["fit"]["weighted_MSE_objective"] / 25
    values["train/anchor_penalty"] = record.get("anchor_penalty", 0.)
    values["param/relative_distance"] = record.get("relative_parameter_distance", 0.)
    if "historical_control_learning_rate" in record:
        values["optim/historical_control_learning_rate"] = record["historical_control_learning_rate"]
        values["optim/LR_differs_from_control"] = int(record["LR_differs_from_control"])
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="CPU-only validation; writes no run files")
    parser.add_argument("--resume", action="store_true", help="Resume only this C148 arm from its complete last epoch")
    args = parser.parse_args()
    plan, base, parent, initial, preflight = read_plan()
    if args.preflight:
        print(json.dumps(preflight, ensure_ascii=False), flush=True)
        return
    if "src/c148_train.py" not in plan["source_sha256"]:
        raise ValueError("Freeze c148_train.py hash in the protocol before launch")
    protocol_sha = preflight["protocol_sha256"]
    if args.resume:
        config = json.loads((RUN / "config.json").read_text())
        if config["protocol_sha256"] != protocol_sha:
            raise ValueError("C148 resume protocol differs from original")
        if (RUN / "training_result.json").exists():
            if json.loads((RUN / "training_result.json").read_text()).get("status") == "complete":
                raise ValueError("C148 is already complete; do not repeat")
        current = torch.load(RUN / "last.pt", map_location="cpu", weights_only=False)
        if current["anchor_identity"]["checkpoint_sha256"] != plan["parent_checkpoint_sha256"]:
            raise ValueError("Resume anchor identity differs")
    else:
        if RUN.exists():
            raise FileExistsError("C148 run directory already exists; inspect it, then use explicit --resume")
        RUN.mkdir(parents=True, exist_ok=False)
        current = parent
    lock = RUN / "active.json"
    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(lock_fd, json.dumps({"pid": os.getpid(), "created_utc": datetime.now(timezone.utc).isoformat()}).encode())
    os.close(lock_fd)
    tracking_started = False
    try:
        device_before = device_snapshot()
        gpu = gpu_guard()
        torch.set_num_threads(6)
        torch.set_float32_matmul_precision("high")
        start = time.monotonic()
        data = reader()
        panels = {name: ResidualPanel(data, base["panels"][name], "context")
                  for name in ("fit", "monitor")}
        actual_panels = {name: {"maps": len(panel.ids), "terrains": len(set(panel.terrain))}
                         for name, panel in panels.items()}
        if any(actual_panels[name] != plan["evaluation_scope"][name] for name in panels):
            raise ValueError("Actual fixed FIT/monitor panel scope differs")
        train = panels["fit"]
        full_fit = np.load(ROOT / "features/c38_dense/fit_map_indices.npy")
        full_count = Counter(data.rows[int(index)]["terrain_id"] for index in full_fit)
        selected_count = Counter(train.terrain)
        weights = torch.tensor([full_count[terrain] / selected_count[terrain]
                                for terrain in train.terrain], device="cuda", dtype=torch.float32)
        weights /= weights.mean()
        batch = base["batch"]
        steps = math.ceil(len(train.ids) / batch)
        if steps != 352:
            raise ValueError("Original C145 steps/epoch changed")
        denominator = ((train.w * weights[:, None, None, None]).sum((0, 2, 3),
                       dtype=torch.float64) / steps).float()
        net = create("native").cuda().to(memory_format=torch.channels_last)
        # The anchor never shares storage with either the model or its optimizer.
        anchor = {name: parent["state"][name].to(device="cuda", dtype=parameter.dtype).detach().clone()
                  for name, parameter in net.named_parameters() if parameter.requires_grad}
        if any(anchor[name].data_ptr() == parameter.data_ptr()
               for name, parameter in net.named_parameters() if parameter.requires_grad):
            raise ValueError("Anchor shares model storage")
        net.load_state_dict(current["state"], strict=True)
        if not args.resume:
            forward_check = verify_parent_forward(net, panels, batch)
        else:
            forward_check = json.loads((RUN / "restoration.json").read_text())["parent_forward_check"]
        optimizer = torch.optim.AdamW(net.parameters(), lr=base["learning_rate"],
                                      weight_decay=base["weight_decay"], fused=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=base["lr_factor"],
            patience=base["lr_patience_evaluations"],
            threshold=base["minimum_relative_fit_improvement"], threshold_mode="rel",
            min_lr=base["minimum_learning_rate"])
        optimizer.load_state_dict(current["optimizer"])
        scheduler.load_state_dict(current["scheduler"])
        state = deepcopy(current["training_state"])
        if state["additional_step"] != state["additional_epoch"] * steps:
            raise ValueError("Full checkpoint update/epoch relationship differs")
        actual_optimizer_steps = {int(value["step"].item()) for value in optimizer.state.values()}
        if actual_optimizer_steps != {state["additional_step"]}:
            raise ValueError("Restored optimizer update count differs")
        if digest(net) != state_hash(current["state"]):
            raise ValueError("Restored model hash differs")
        generator = torch.Generator()
        history = deepcopy(current["history"])
        history[0]["initial_metrics_reused"] = True
        target_epoch = plan["parent_additional_epoch"] + plan["additional_epochs"]
        if state["additional_epoch"] >= target_epoch:
            raise ValueError("Checkpoint is already at or beyond the fixed C148 budget")
        historical = {row["absolute_epoch"]: row for row in json.loads((PARENT_RUN / "history.json").read_text())}
        config = {"campaign": "C148", "protocol_sha256": protocol_sha, "protocol": plan,
                  "parent_absolute_epoch": 12, "additional_budget": 32, "panels": base["panels"],
                  "actual_panels": actual_panels,
                  "GPU": gpu, "launch_device_snapshot": device_before,
                  "diagnosis_path": "audit/c148_generalization/diagnosis.json",
                  "supervision": base["loss"], "official_VAL_used": False,
                  "_cc_dashboard": {"scope": "同一开发 monitor：128地形628图；80地形曾用于C82校准",
                      "baseline_source": "audit/c145_capacity/comparison_summary.json",
                      "C82": C82, "step_unit": "epoch", "total_epochs": 44,
                      "comparison_contract": "Same checkpoint, both bands; must beat selected12 and C82. Reused development monitor, not fresh blind."}}
        if not args.resume:
            atomic_json(RUN / "config.json", config)
        previous_tracking = json.loads((RUN / "tracking.json").read_text()) if args.resume else {}
        linked_id = previous_tracking.get("run_id", PARENT_SWANLAB_ID)
        tracking_run = cc_tracking.init(
            name="C148 selected12 parameter anchor" + (" resume" if args.resume else ""),
            config=config, campaign="C148", parent_run_id=linked_id,
            parent_checkpoint=str((RUN / "last.pt") if args.resume else ROOT / plan["parent_checkpoint"]),
            start_step=state["additional_epoch"] + 8,
            description="L2-SP lambda=.1; one fixed 32-epoch arm; historical C145 control reused; local only.",
            tags=["parameter-anchor", "development-monitor", "no-official-VAL"])
        tracking_started = True
        atomic_json(RUN / "tracking.json", {"run_id": getattr(tracking_run, "id", None),
                    "dir": str(getattr(tracking_run, "dir", "")), "parent_run_id": linked_id,
                    "step_unit": "epoch", "start_step": state["additional_epoch"] + 8})
        # All initialization and the one-batch inference audit are complete.
        # Restore all recorded RNGs last, before the first continuation permutation.
        rng = restore_rng(current, generator)
        restoration = {**preflight, "resume": args.resume, "restored_optimizer_steps": sorted(actual_optimizer_steps),
                       "restored_additional_epoch": state["additional_epoch"],
                       "restored_additional_step": state["additional_step"],
                       "restored_rng_sha256": rng, "parent_forward_check": forward_check,
                       "scheduler_input": "Original full weighted FIT data MSE; excludes L2-SP",
                       "Python_NumPy_RNG": "Not in C145 checkpoint; audited training order uses restored torch.Generator"}
        atomic_json(RUN / "restoration.json", restoration)
        anchor_identity = {"checkpoint": plan["parent_checkpoint"],
                           "checkpoint_sha256": plan["parent_checkpoint_sha256"],
                           "model_sha256": plan["anchor_model_sha256"],
                           "norm_sq": plan["anchor_norm_sq"], "l2sp_lambda": plan["l2sp_lambda"]}

        def checkpoint():
            result = checkpoint_data(net, optimizer, scheduler, generator, deepcopy(state), deepcopy(history))
            result["anchor_identity"] = anchor_identity
            result["c148_protocol_sha256"] = protocol_sha
            result["tracking"] = json.loads((RUN / "tracking.json").read_text())
            return result

        phase_start = state["additional_epoch"]
        if not args.resume:
            atomic_json(RUN / "history.json", history)
            save_checkpoint(RUN / "parent.pt", checkpoint())
            save_checkpoint(RUN / "best_fit.pt", checkpoint())
            save_checkpoint(RUN / "best_monitor.pt", checkpoint())
            save_checkpoint(RUN / "last.pt", checkpoint())
            cc_tracking.log(metric_log(initial, 0., 0.), step=12)
        for epoch in range(phase_start + 1, target_epoch + 1):
            order = torch.randperm(len(train.ids), generator=generator)
            totals = torch.zeros(4, device="cuda", dtype=torch.float64)
            sums = torch.zeros(3, device="cuda", dtype=torch.float64)
            gradient_total = 0.
            clipped = 0
            net.train()
            learning_rate_used = optimizer.param_groups[0]["lr"]
            progress = tqdm(enumerate(order.split(batch)), total=steps,
                            desc=f"C148 Epoch {epoch+8}/44", mininterval=5)
            for index, positions in progress:
                if state["additional_step"] < base["warmup_updates"]:
                    fraction = (state["additional_step"] + 1) / base["warmup_updates"]
                    for group in optimizer.param_groups:
                        group["lr"] = base["learning_rate"] * (.1 + .9 * fraction)
                ix = positions.cuda()
                x, baseline, target, valid = train.batch(ix)
                with torch.autocast("cuda", torch.bfloat16):
                    prediction = baseline + net(x).float()
                error = torch.where(valid, prediction - target, 0.)
                inclusion = weights[ix, None, None, None]
                numerator = (error.square() * inclusion).sum((0, 2, 3))
                data_loss = (numerator / denominator).mean() / 25
                relative_distance = parameter_distance(net, anchor) / plan["anchor_norm_sq"]
                penalty = plan["l2sp_lambda"] * relative_distance
                loss = data_loss + penalty
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite C148 training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite C148 gradient")
                optimizer.step()
                state["additional_step"] += 1
                gradient_total += float(norm)
                clipped += int(norm > 2)
                totals += torch.cat((numerator.detach(), (valid * inclusion).sum((0, 2, 3))))
                sums += torch.stack((data_loss.detach(), penalty.detach(), relative_distance.detach())).double()
                if (index + 1) % 40 == 0 or index + 1 == steps:
                    elapsed = time.monotonic() - start
                    done = epoch - phase_start - 1 + (index + 1) / steps
                    eta = elapsed / done * (target_epoch - phase_start - done)
                    atomic_json(RUN / "progress.json", {"stage": "training",
                        "absolute_epoch": epoch + 8, "target_absolute_epoch": 44,
                        "additional_epoch": epoch, "additional_step": state["additional_step"],
                        "batches_done": index + 1, "batches_total": steps,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "elapsed_phase_seconds": elapsed, "eta_seconds": max(0., eta)})
                    progress.set_postfix(data=f"{float(data_loss):.4f}", anchor=f"{float(penalty):.4f}")
            state["additional_epoch"] = epoch
            if epoch % base["evaluate_every_epochs"] == 0 or epoch == target_epoch:
                metrics = {
                    "fit": evaluate(net, train, batch, f"C148 FIT Epoch {epoch+8}", map_weights=weights),
                    "monitor": evaluate(net, panels["monitor"], batch, f"C148 monitor Epoch {epoch+8}")}
                objective = metrics["fit"]["weighted_MSE_objective"]
                mean_monitor = float(np.mean(metrics["monitor"]["rmse"]))
                fit_improved = objective < state["best_fit_objective"]
                monitor_improved = mean_monitor < state["best_monitor_mean_rmse"]
                if fit_improved:
                    state["best_fit_objective"] = objective
                    state["best_fit_epoch"] = epoch
                if monitor_improved:
                    state["best_monitor_mean_rmse"] = mean_monitor
                    state["best_monitor_epoch"] = epoch
                if objective < state["meaningful_fit_best"] * (1 - base["minimum_relative_fit_improvement"]):
                    state["meaningful_fit_best"] = objective
                    state["last_meaningful_fit_improvement_epoch"] = epoch
                old_rate = optimizer.param_groups[0]["lr"]
                scheduler.step(objective)
                rate = optimizer.param_groups[0]["lr"]
                if rate < old_rate * (1 - 1e-8):
                    state["learning_rate_reductions"] += 1
                    state["last_lr_reduction_epoch"] = epoch
                relative = float(parameter_distance(net, anchor).detach() / plan["anchor_norm_sq"])
                elapsed = time.monotonic() - start
                eta = elapsed / (epoch - phase_start) * (target_epoch - epoch)
                control_rate = historical[epoch + 8]["learning_rate"]
                record = {"additional_epoch": epoch, "absolute_epoch": epoch + 8,
                    "additional_step": state["additional_step"], "evaluation": metrics,
                    "running_weighted_RMSE": (totals[:2] / totals[2:]).sqrt().cpu().tolist(),
                    "running_data_loss": float(sums[0] / steps),
                    "running_anchor_penalty": float(sums[1] / steps),
                    "relative_parameter_distance": relative, "anchor_penalty": plan["l2sp_lambda"] * relative,
                    "mean_gradient_norm": gradient_total / steps, "clipped_batch_fraction": clipped / steps,
                    "learning_rate_used": learning_rate_used, "learning_rate": rate,
                    "historical_control_learning_rate": control_rate,
                    "LR_differs_from_control": not math.isclose(rate, control_rate, abs_tol=1e-15, rel_tol=0),
                    "learning_rate_reductions": state["learning_rate_reductions"],
                    "last_lr_reduction_epoch": state["last_lr_reduction_epoch"],
                    "best_fit_epoch": state["best_fit_epoch"], "best_monitor_epoch": state["best_monitor_epoch"],
                    "peak_GPU_GiB": torch.cuda.max_memory_allocated() / 2**30,
                    "elapsed_phase_seconds": elapsed, "eta_seconds": eta,
                    "stop_reason": None, "official_VAL_used": False}
                history.append(record)
                atomic_json(RUN / "history.json", history)
                atomic_json(RUN / "progress.json", {"stage": "evaluated", **record})
                cc_tracking.log(metric_log(record, elapsed, eta), step=epoch + 8)
                if fit_improved:
                    save_checkpoint(RUN / "best_fit.pt", checkpoint())
                if monitor_improved:
                    save_checkpoint(RUN / "best_monitor.pt", checkpoint())
                print(json.dumps({"event": "evaluation", **record}), flush=True)
            save_checkpoint(RUN / "last.pt", checkpoint())
        selected = next(record for record in history
                        if record["additional_epoch"] == state["best_monitor_epoch"])
        selected_rmse = selected["evaluation"]["monitor"]["rmse"]
        parent_rmse = initial["evaluation"]["monitor"]["rmse"]
        result = {"status": "complete", "training_state": state,
            "parent_absolute_epoch": 12, "additional_budget": 32, "absolute_epoch": 44,
            "selected_absolute_epoch": state["best_monitor_epoch"] + 8,
            "selected_monitor_RMSE_dB": selected_rmse,
            "improves_both_monitor_bands_over_parent_and_C82":
                all(selected_rmse[i] < min(parent_rmse[i], C82["monitor"][i]) for i in range(2)),
            "selected_remains_parent": state["best_monitor_epoch"] == 4,
            "elapsed_phase_seconds": time.monotonic() - start, "official_VAL_used": False,
            "scope": plan["evaluation_scope"], "protocol_sha256": protocol_sha,
            "control_rerun": False, "fixed_budget_complete": True}
        atomic_json(RUN / "training_result.json", result)
        print(json.dumps({"event": "complete", **result}), flush=True)
    finally:
        if tracking_started:
            cc_tracking.finish()
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
