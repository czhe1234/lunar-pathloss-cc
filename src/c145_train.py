"""C145 sufficient-training capability check of C82 plus the C123 native branch."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cc_core import atomic_json, check_source_unchanged, gpu_guard
from src.c105_context import ResidualPanel, digest, reader
from src.c123_lossless import create
import numpy as np
import torch
from tqdm import tqdm


AUDIT = ROOT / "audit/c145_capacity"
RUN = ROOT / "runs/c145_capacity_native"
GROUPS = ("all", "tail", "under_tail", "over_tail", "non_tail", "C82_good")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_checkpoint(path, data):
    temporary = path.with_suffix(".partial.pt")
    torch.save(data, temporary)
    temporary.replace(path)


@torch.inference_mode()
def evaluate(net, panel, batch, label, map_weights=None, predictions=None):
    net.eval()
    sse = torch.zeros((2, len(GROUPS), 2), dtype=torch.float64, device="cuda")
    counts = torch.zeros((len(GROUPS), 2), dtype=torch.int64, device="cuda")
    weighted_sse = torch.zeros(2, dtype=torch.float64, device="cuda")
    weighted_count = torch.zeros_like(weighted_sse)
    new_errors = torch.zeros((2, 2), dtype=torch.int64, device="cuda")
    tail_sign = torch.zeros(2, dtype=torch.int64, device="cuda")
    total = math.ceil(len(panel.ids) / batch)
    progress = tqdm(range(0, len(panel.ids), batch), total=total, desc=label, mininterval=5)
    for start in progress:
        stop = min(start + batch, len(panel.ids))
        selection = slice(start, stop)
        x, base, target, valid = panel.batch(selection)
        with torch.autocast("cuda", torch.bfloat16):
            prediction = base + net(x).float()
        if not torch.isfinite(prediction).all():
            raise RuntimeError("Nonfinite prediction during full-panel evaluation")
        error = prediction.double() - target.double()
        residual = target.double() - base.double()
        tail = residual.abs() >= 5
        masks = (
            valid,
            valid & tail,
            valid & (residual >= 5),
            valid & (residual <= -5),
            valid & ~tail,
            valid & (residual.abs() <= 1),
        )
        candidate_square = error.square()
        baseline_square = residual.square()
        for group, mask in enumerate(masks):
            sse[0, group] += torch.where(mask, candidate_square, 0).sum((0, 2, 3))
            sse[1, group] += torch.where(mask, baseline_square, 0).sum((0, 2, 3))
            counts[group] += mask.sum((0, 2, 3))
        for index, threshold in enumerate((3, 5)):
            new_errors[index] += (masks[-1] & (error.abs() > threshold)).sum((0, 2, 3))
        correction = prediction.double() - base.double()
        tail_sign += (masks[1] & (correction * residual > 0)).sum((0, 2, 3))
        if map_weights is not None:
            weights = map_weights[selection, None, None, None].double()
            weighted_sse += (torch.where(valid, candidate_square, 0) * weights).sum((0, 2, 3))
            weighted_count += (valid * weights).sum((0, 2, 3))
        if predictions is not None:
            predictions[start:stop] = prediction.cpu().numpy()
    if predictions is not None:
        predictions.flush()
    ss = sse.cpu().numpy()
    cc = counts.cpu().numpy()
    rmse = np.sqrt(np.divide(ss, cc[None], out=np.zeros_like(ss), where=cc[None] > 0))
    result = {
        "groups": {
            name: {
                "RMSE": rmse[0, index].tolist(),
                "C82_RMSE": rmse[1, index].tolist(),
                "SSE": ss[0, index].tolist(),
                "C82_SSE": ss[1, index].tolist(),
                "count": cc[index].tolist(),
            }
            for index, name in enumerate(GROUPS)
        },
        "tail_correction_direction_accuracy": (
            tail_sign.cpu().numpy() / np.maximum(cc[1], 1)
        ).tolist(),
        "C82_good_new_error_above_3dB_count": new_errors[0].cpu().tolist(),
        "C82_good_new_error_above_5dB_count": new_errors[1].cpu().tolist(),
    }
    result["rmse"] = result["groups"]["all"]["RMSE"]
    result["C82_rmse"] = result["groups"]["all"]["C82_RMSE"]
    if map_weights is not None:
        ws, wc = weighted_sse.cpu().numpy(), weighted_count.cpu().numpy()
        result["weighted_RMSE"] = np.sqrt(ws / wc).tolist()
        result["weighted_MSE_objective"] = float(np.mean(ws / wc))
        result["weighted_valid_count"] = wc.tolist()
    net.train()
    return result


def checkpoint_data(net, optimizer, scheduler, order_generator, state, history):
    return {
        "state": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "data_generator": order_generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "training_state": state,
        "history": history,
    }


def prediction_identity(model_digest, panel):
    ids = np.asarray(panel.ids, dtype="<i8")
    return {
        "model_sha256": model_digest,
        "panel_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "shape": [len(panel.ids), 2, 256, 256],
        "prediction_dtype": "float32",
        "forward_precision": "BF16 autocast",
    }


def prediction_sidecar(path):
    return path.with_suffix(".json")


def cached_prediction_metadata(path, identity):
    sidecar = prediction_sidecar(path)
    if not path.is_file() or not sidecar.is_file():
        return None
    try:
        metadata = json.loads(sidecar.read_text())
        if any(metadata.get(key) != value for key, value in identity.items()):
            return None
        if metadata.get("file_size_bytes") != path.stat().st_size:
            return None
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        valid_layout = list(array.shape) == identity["shape"] and array.dtype == np.float32
        del array
        if not valid_layout or metadata.get("prediction_sha256") != file_hash(path):
            return None
        return metadata
    except (OSError, ValueError, KeyError, TypeError):
        return None


def record_prediction(path, identity, linked_metadata=None, linked_from=None):
    if linked_metadata is None:
        metadata = {
            **identity, "file_size_bytes": path.stat().st_size,
            "prediction_sha256": file_hash(path),
        }
    else:
        metadata = {**identity, "file_size_bytes": path.stat().st_size,
                    "prediction_sha256": linked_metadata["prediction_sha256"],
                    "linked_from": linked_from}
    atomic_json(prediction_sidecar(path), metadata)
    return metadata


def export_checkpoints(net, panels, batch):
    exported = {}
    for tag, filename in (
        ("selected", "best_monitor.pt"),
        ("fit_best", "best_fit.pt"),
        ("last", "last.pt"),
    ):
        path = RUN / filename
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        net.load_state_dict(checkpoint["state"])
        model_digest = digest(net)
        metadata = {
            "tag": tag, "checkpoint": filename, "model_sha256": model_digest,
            "training_state": checkpoint["training_state"],
            "prediction_dtype": "float32", "forward_precision": "BF16 autocast",
            "panels": {},
        }
        previous = exported.get(model_digest)
        same_as_previous = previous is not None
        for name, panel in panels.items():
            identity = prediction_identity(model_digest, panel)
            target = RUN / f"{tag}_{name}_prediction.npy"
            completed = cached_prediction_metadata(target, identity)
            source = RUN / f"{previous}_{name}_prediction.npy" if previous else None
            source_metadata = cached_prediction_metadata(source, identity) if source else None
            if completed is None:
                temporary = target.with_suffix(".partial.npy")
                if temporary.exists():
                    temporary.unlink()
                if source_metadata is not None:
                    os.link(source, temporary)
                    temporary.replace(target)
                    completed = record_prediction(
                        target, identity, linked_metadata=source_metadata,
                        linked_from=str(source.relative_to(RUN)),
                    )
                else:
                    array = np.lib.format.open_memmap(
                        temporary, mode="w+", dtype=np.float32, shape=tuple(identity["shape"])
                    )
                    evaluate(net, panel, batch, f"C145 export {tag} {name}", predictions=array)
                    del array
                    temporary.replace(target)
                    completed = record_prediction(target, identity)
            metadata["panels"][name] = {"file": target.name, **completed}
            same_as_previous &= (
                source_metadata is not None
                and source_metadata["prediction_sha256"] == completed["prediction_sha256"]
            )
        if same_as_previous:
            metadata["same_predictions_as"] = previous
        atomic_json(RUN / f"{tag}_export.json", metadata)
        exported.setdefault(model_digest, tag)


def reconcile_history(history, saved_epoch, saved_step):
    if not isinstance(history, list):
        raise ValueError("C145 evaluation history must be a list")
    retained = []
    previous_epoch = 0
    previous_step = 0
    for record in history:
        epoch = record["additional_epoch"]
        step = record["additional_step"]
        if not isinstance(epoch, int) or not isinstance(step, int):
            raise ValueError("C145 history epochs and steps must be integers")
        if epoch > saved_epoch:
            continue
        if epoch <= previous_epoch or step <= previous_step or step > saved_step:
            raise ValueError("C145 history is inconsistent with its checkpoint")
        retained.append(record)
        previous_epoch = epoch
        previous_step = step
    return retained


def validate_resume_configuration(saved, plan):
    # Phase lengths and recorded source versions may change during a logged
    # recovery. The data, model, optimizer, schedule and stopping fields stay fixed.
    noncausal = {
        "created", "source_sha256", "initial_phase_additional_epochs",
        "extension_epochs_if_still_learning",
    }
    mismatches = [key for key, value in plan.items()
                  if key not in noncausal and saved.get(key) != value]
    if mismatches:
        raise ValueError("C145 fixed configuration changed: " + ", ".join(mismatches))


def load_resume_checkpoint():
    latest = None
    latest_path = None
    for filename in ("last.pt", "best_fit.pt", "best_monitor.pt"):
        path = RUN / filename
        if not path.is_file():
            continue
        candidate = torch.load(path, map_location="cpu", weights_only=False)
        epoch = candidate["training_state"]["additional_epoch"]
        if latest is None or epoch > latest["training_state"]["additional_epoch"]:
            latest, latest_path = candidate, path
    if latest is None:
        raise FileNotFoundError("No complete C145 checkpoint is available")
    state = latest["training_state"]
    history = latest.get("history")
    if history is None:
        history = json.loads((RUN / "history.json").read_text())
    history = reconcile_history(history, state["additional_epoch"], state["additional_step"])
    atomic_json(RUN / "history.json", history)
    # A best checkpoint contains the full recovery state. This also repairs a
    # crash between saving a newly best checkpoint and updating last.pt.
    if latest_path.name != "last.pt":
        save_checkpoint(RUN / "last.pt", latest)
    for filename, epoch_key in (
        ("best_fit.pt", "best_fit_epoch"), ("best_monitor.pt", "best_monitor_epoch")
    ):
        expected_epoch = state[epoch_key]
        if expected_epoch == state["additional_epoch"]:
            save_checkpoint(RUN / filename, latest)
            continue
        path = RUN / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing C145 selected checkpoint: {filename}")
        selected = torch.load(path, map_location="cpu", weights_only=False)
        if selected["training_state"]["additional_epoch"] != expected_epoch:
            raise ValueError(f"C145 selected checkpoint epoch differs: {filename}")
    return latest, history, latest_path.name


def plateau_stop_ready(state, epoch, plan):
    return (
        epoch >= plan["minimum_additional_epochs"]
        and state["learning_rate_reductions"] >= plan["minimum_lr_reductions_for_stop"]
        and epoch-state["last_meaningful_fit_improvement_epoch"] >= plan["plateau_epochs"]
        and epoch-state["last_lr_reduction_epoch"] >= plan["epochs_after_last_lr_reduction"]
    )


def begin_phase(gpu, args, state, checkpoint_used):
    path = RUN / "phases.json"
    phases = json.loads(path.read_text()) if path.exists() else []
    record = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "GPU": gpu, "starting_additional_epoch": state["additional_epoch"],
        "phase_cap_additional_epoch": args.max_epochs,
        "export_only": args.export_only, "checkpoint_used": checkpoint_used,
        "source_sha256": file_hash(Path(__file__)),
        "protocol_sha256": file_hash(AUDIT / "protocol.json"), "status": "running",
    }
    phases.append(record)
    atomic_json(path, phases)
    return phases


def finish_phase(phases, status, state, elapsed):
    phases[-1].update({
        "finished_utc": datetime.now(timezone.utc).isoformat(), "status": status,
        "ending_additional_epoch": state["additional_epoch"], "elapsed_seconds": elapsed,
    })
    atomic_json(RUN / "phases.json", phases)


def record_completed_export(state, elapsed):
    path = RUN / "training_result.json"
    prior = json.loads(path.read_text()) if path.exists() else {}
    if prior.get("training_state", {}).get("additional_epoch") == state["additional_epoch"]:
        result = {**prior, "status": "training_complete", "training_state": state}
    else:
        result = {
            "status": "training_complete", "training_state": state,
            "phase_end_epoch": state["additional_epoch"],
            "elapsed_phase_seconds": elapsed, "official_VAL_used": False,
            "recovered_after_checkpoint_commit": True,
        }
    atomic_json(path, result)
    atomic_json(RUN / "result.json", {**result, "status": "complete_pending_independent_scoring"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=128,
                        help="Absolute number of additional C145 epochs in this phase")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true",
                        help="Explicit C145 user authorization for shared physical card 3 only")
    args = parser.parse_args()
    plan = json.loads((AUDIT / "protocol.json").read_text())
    parent = ROOT / plan["parent_checkpoint"]
    if file_hash(parent) != plan["parent_checkpoint_sha256"]:
        raise ValueError("C145 parent checkpoint changed")
    for name, expected in plan["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"C145 source differs from the recorded protocol: {name}")
    if args.resume or args.export_only:
        validate_resume_configuration(json.loads((RUN / "config.json").read_text()), plan)
    # Imports and file checks can outlast a short vacant GPU window.
    # Recheck immediately before the first CUDA initialization.
    identifier = os.environ["CC_GPU_UUID"]
    if args.allow_shared_gpu:
        authorization = json.loads((AUDIT / "sharing_authorization.json").read_text())
        approved_shared_uuid = "GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0"
        if (identifier != approved_shared_uuid or authorization.get("scope") != "C145 only"
                or authorization.get("gpu_uuid") != identifier):
            raise ValueError("Shared GPU permission is restricted to the explicitly authorized card 3")
    jobs = subprocess.check_output(
        ["nvidia-smi", "-i", identifier, "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"], text=True, timeout=10,
    ).strip()
    device_stats = subprocess.check_output(
        ["nvidia-smi", "-i", identifier, "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=10,
    ).strip().split(",")
    if not args.allow_shared_gpu and (jobs or int(device_stats[0]) > 256 or int(device_stats[1]) != 0):
        print("Selected GPU became busy before CUDA initialization; launch deferred.", flush=True)
        raise SystemExit(75)
    gpu = gpu_guard()
    gpu.update({
        "sharing_authorized": args.allow_shared_gpu,
        "co_resident_pids_at_launch": jobs.splitlines() if jobs else [],
        "memory_used_before_initialization_MiB": int(device_stats[0]),
        "utilization_before_initialization_percent": int(device_stats[1]),
    })
    print(json.dumps({"event": "GPU_ready", "GPU": gpu}), flush=True)
    check_source_unchanged()
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("high")
    start = time.monotonic()
    data = reader()
    panels = {
        name: ResidualPanel(data, plan["panels"][name], "context")
        for name in ("fit", "monitor")
    }
    train = panels["fit"]
    full_fit = np.load(ROOT / "features/c38_dense/fit_map_indices.npy")
    full_count = Counter(data.rows[int(index)]["terrain_id"] for index in full_fit)
    selected_count = Counter(train.terrain)
    weights = torch.tensor(
        [full_count[terrain] / selected_count[terrain] for terrain in train.terrain],
        device="cuda", dtype=torch.float32,
    )
    weights /= weights.mean()
    batch = plan["batch"]
    steps = math.ceil(len(train.ids) / batch)
    global_den = (train.w * weights[:, None, None, None]).sum((0, 2, 3), dtype=torch.float64)
    denominator = (global_den / steps).float()
    net = create("native").cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=plan["learning_rate"], weight_decay=plan["weight_decay"], fused=True
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=plan["lr_factor"], patience=plan["lr_patience_evaluations"],
        threshold=plan["minimum_relative_fit_improvement"], threshold_mode="rel",
        min_lr=plan["minimum_learning_rate"],
    )
    generator = torch.Generator().manual_seed(plan["data_seed"])
    if args.resume or args.export_only:
        checkpoint, history, checkpoint_used = load_resume_checkpoint()
        net.load_state_dict(checkpoint["state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        generator.set_state(checkpoint["data_generator"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"])
        state = checkpoint["training_state"]
        if state["additional_step"] != state["additional_epoch"]*steps:
            raise ValueError("C145 checkpoint step count differs from its fixed epoch size")
        saved_config = json.loads((RUN / "config.json").read_text())
        actual_panels = {
            name: {"maps": len(panel.ids), "terrains": len(set(panel.terrain))}
            for name, panel in panels.items()
        }
        if saved_config["actual_panels"] != actual_panels:
            raise ValueError("C145 resumed panels differ from the original run")
    else:
        RUN.mkdir(exist_ok=False)
        parent = ROOT / plan["parent_checkpoint"]
        assert file_hash(parent) == plan["parent_checkpoint_sha256"]
        checkpoint = torch.load(parent, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 8
        assert "optimizer" not in checkpoint
        net.load_state_dict(checkpoint["state"])
        # Advance the documented C123 generator past the eight completed permutations.
        for _ in range(8):
            torch.randperm(len(train.ids), generator=generator)
        config = {
            **plan, "GPU": gpu, "parameters": sum(p.numel() for p in net.parameters()),
            "actual_panels": {
                name: {"maps": len(panel.ids), "terrains": len(set(panel.terrain))}
                for name, panel in panels.items()
            },
            "actual_training_weighted_valid_count": global_den.cpu().tolist(),
            "initial_model_state_sha256": digest(net),
        }
        atomic_json(RUN / "config.json", config)
        initial = {
            "fit": evaluate(net, train, batch, "C145 initial FIT", map_weights=weights),
            "monitor": evaluate(net, panels["monitor"], batch, "C145 initial monitor"),
        }
        reference = np.array([2.146390507442155, 2.4180144850568968])
        assert np.max(np.abs(np.array(initial["monitor"]["C82_rmse"])-reference)) < 2e-5
        expected = np.array([2.145354151487308, 2.4170135345427792])
        assert np.max(np.abs(np.array(initial["monitor"]["rmse"])-expected)) < .003
        atomic_json(RUN / "initial.json", initial)
        state = {
            "additional_epoch": 0, "additional_step": 0,
            "best_fit_objective": initial["fit"]["weighted_MSE_objective"],
            "best_fit_epoch": 0, "best_monitor_mean_rmse": float(np.mean(initial["monitor"]["rmse"])),
            "best_monitor_epoch": 0, "meaningful_fit_best": initial["fit"]["weighted_MSE_objective"],
            "last_meaningful_fit_improvement_epoch": 0, "learning_rate_reductions": 0,
            "last_lr_reduction_epoch": 0, "stop_reason": None,
        }
        scheduler.step(initial["fit"]["weighted_MSE_objective"])
        history = []
        atomic_json(RUN / "history.json", history)
        checkpoint = checkpoint_data(net, optimizer, scheduler, generator, state, history)
        for name in ("last.pt", "best_fit.pt", "best_monitor.pt"):
            save_checkpoint(RUN / name, checkpoint)
        print(json.dumps({"event": "initial", "fit": initial["fit"], "monitor": initial["monitor"],
                          "loaded_seconds": time.monotonic()-start}), flush=True)
        checkpoint_used = plan["parent_checkpoint"]
    phases = begin_phase(gpu, args, state, checkpoint_used)
    if args.export_only:
        export_checkpoints(net, panels, batch)
        if state["stop_reason"]:
            record_completed_export(state, time.monotonic()-start)
        finish_phase(phases, "export_complete", state, time.monotonic()-start)
        return
    if state["stop_reason"]:
        print(json.dumps({"event": "already_stopped", "reason": state["stop_reason"]}), flush=True)
        export_checkpoints(net, panels, batch)
        record_completed_export(state, time.monotonic()-start)
        finish_phase(phases, "already_stopped_export_complete", state, time.monotonic()-start)
        return
    phase_start = state["additional_epoch"]
    if args.max_epochs <= phase_start:
        raise ValueError("Phase cap must exceed the saved additional epoch")
    for epoch in range(phase_start+1, args.max_epochs+1):
        order = torch.randperm(len(train.ids), generator=generator)
        totals = torch.zeros(4, device="cuda", dtype=torch.float64)
        gradient_total = 0.
        clipped = 0
        net.train()
        batches = order.split(batch)
        progress = tqdm(enumerate(batches), total=steps,
                        desc=f"C145 +epoch {epoch}/{args.max_epochs}", mininterval=5)
        for index, positions in progress:
            if state["additional_step"] < plan["warmup_updates"]:
                fraction = (state["additional_step"]+1)/plan["warmup_updates"]
                rate = plan["learning_rate"]*(.1+.9*fraction)
                for group in optimizer.param_groups:
                    group["lr"] = rate
            ix = positions.cuda()
            x, baseline, target, valid = train.batch(ix)
            with torch.autocast("cuda", torch.bfloat16):
                prediction = baseline + net(x).float()
            error = torch.where(valid, prediction-target, 0.)
            inclusion = weights[ix, None, None, None]
            numerator = (error.square()*inclusion).sum((0, 2, 3))
            loss = (numerator/denominator).mean()/25
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite gradient")
            optimizer.step()
            state["additional_step"] += 1
            gradient_total += float(norm)
            clipped += int(norm > 2)
            valid_count = (valid*inclusion).sum((0, 2, 3))
            totals += torch.cat((numerator.detach(), valid_count))
            if (index+1) % 40 == 0 or index+1 == steps:
                atomic_json(RUN / "progress.json", {
                    "stage": "training", "additional_epoch": epoch,
                    "absolute_epoch": epoch+8, "batches_done": index+1, "batches_total": steps,
                    "additional_step": state["additional_step"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_phase_seconds": time.monotonic()-start,
                })
        state["additional_epoch"] = epoch
        evaluated = epoch % plan["evaluate_every_epochs"] == 0 or epoch == args.max_epochs
        if evaluated:
            metrics = {
                "fit": evaluate(net, train, batch, f"C145 FIT +epoch {epoch}", map_weights=weights),
                "monitor": evaluate(net, panels["monitor"], batch, f"C145 monitor +epoch {epoch}"),
            }
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
            if objective < state["meaningful_fit_best"]*(1-plan["minimum_relative_fit_improvement"]):
                state["meaningful_fit_best"] = objective
                state["last_meaningful_fit_improvement_epoch"] = epoch
            old_rate = optimizer.param_groups[0]["lr"]
            scheduler.step(objective)
            if optimizer.param_groups[0]["lr"] < old_rate*(1-1e-8):
                state["learning_rate_reductions"] += 1
                state["last_lr_reduction_epoch"] = epoch
            plateau = epoch-state["last_meaningful_fit_improvement_epoch"]
            if plateau_stop_ready(state, epoch, plan):
                state["stop_reason"] = "weighted_full_FIT_plateau_after_settled_learning_rate_reductions"
            record = {
                "additional_epoch": epoch, "absolute_epoch": epoch+8,
                "additional_step": state["additional_step"], "evaluation": metrics,
                "running_weighted_RMSE": (totals[:2]/totals[2:]).sqrt().cpu().tolist(),
                "mean_gradient_norm": gradient_total/steps, "clipped_batch_fraction": clipped/steps,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "learning_rate_reductions": state["learning_rate_reductions"],
                "last_lr_reduction_epoch": state["last_lr_reduction_epoch"],
                "epochs_since_last_lr_reduction": epoch-state["last_lr_reduction_epoch"],
                "best_fit_epoch": state["best_fit_epoch"],
                "best_monitor_epoch": state["best_monitor_epoch"],
                "epochs_since_meaningful_FIT_improvement": plateau,
                "peak_GPU_GiB": torch.cuda.max_memory_allocated()/2**30,
                "elapsed_phase_seconds": time.monotonic()-start,
                "stop_reason": state["stop_reason"],
            }
            history.append(record)
            atomic_json(RUN / "history.json", history)
            atomic_json(RUN / "progress.json", {"stage": "evaluated", **record})
            checkpoint = checkpoint_data(net, optimizer, scheduler, generator, state, history)
            if fit_improved:
                save_checkpoint(RUN / "best_fit.pt", checkpoint)
            if monitor_improved:
                save_checkpoint(RUN / "best_monitor.pt", checkpoint)
            print(json.dumps({"event": "evaluation", **record}), flush=True)
        save_checkpoint(RUN / "last.pt", checkpoint_data(net, optimizer, scheduler, generator, state, history))
        if state["stop_reason"]:
            break
    result = {
        "status": "training_complete" if state["stop_reason"] else "phase_complete_more_training_needed",
        "training_state": state, "phase_start_epoch": phase_start,
        "phase_end_epoch": state["additional_epoch"],
        "elapsed_phase_seconds": time.monotonic()-start,
        "official_VAL_used": False,
    }
    atomic_json(RUN / "training_result.json", result)
    print(json.dumps({"event": "phase_end", **result}), flush=True)
    finish_phase(phases, result["status"], state, time.monotonic()-start)
    if state["stop_reason"]:
        export_checkpoints(net, panels, batch)
        atomic_json(RUN / "result.json", {**result, "status": "complete_pending_independent_scoring"})


if __name__ == "__main__":
    main()
