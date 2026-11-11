"""C142 fixed-input LLVM/CUDA comparison for the C141 direct-power cases."""
import argparse
import csv
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit/c142_backend"
C141 = ROOT / "audit/c141_direct_receiver"
INPUTS = AUDIT / "inputs"
N = 4_000_000_000
BATCH = 500_000
FREQ = 415e6
POWER_SCALE = (4 * np.pi * FREQ / 299792458.) ** 2
APPROVED = {
    "GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009",
    "GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0",
    "GPU-70b6fceb-c7c4-9467-3c81-097c9284c160",
}


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(".partial.json")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def rt_environment(backend):
    if backend == "llvm":
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide all GPUs for LLVM"
        device = None
    else:
        requested = os.environ.get("CC_GPU_UUID")
        assert requested in APPROVED and os.environ.get("CUDA_VISIBLE_DEVICES") == requested
        driver = ctypes.CDLL("libcuda.so.1")
        assert driver.cuInit(0) == 0, "CUDA driver initialization failed"
        count = ctypes.c_int()
        logical = ctypes.c_int()
        raw = (ctypes.c_ubyte * 16)()
        assert driver.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value == 1
        assert driver.cuDeviceGet(ctypes.byref(logical), 0) == 0
        assert driver.cuDeviceGetUuid(ctypes.byref(raw), logical) == 0
        device = "GPU-" + str(uuid.UUID(bytes=bytes(raw)))
        assert device == requested, (device, requested)
    import mitsuba as mi
    mi.set_variant(f"{backend}_ad_mono_polarized")
    import drjit as dr
    import sionna.rt as rt
    assert (mi.__version__, dr.__version__, rt.__version__) == ("3.6.2", "1.0.3", "1.1.0")
    dr.set_thread_count(2)
    if backend == "cuda":
        dr.eval(dr.arange(mi.Float, 2))
        dr.sync_thread()
    return mi, dr, rt, device


def fixed_panel():
    with (C141 / "remaining_finite_counterexamples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    cases = {}
    for row in rows:
        assert min(float(row[key]) for key in ("violation_415", "violation_58")) > 1
        cases.setdefault(int(row["map"]), []).append(256 * int(row["row"]) + int(row["col"]))
    panel = json.loads((C141 / "input_panel.json").read_text())["records"]
    records = []
    for rec in panel:
        mid = rec["map"]
        if mid in cases:
            assert set(cases[mid]) <= set(rec["pixels"][:-8])
            records.append({"map": mid, "case_pixels": cases[mid], "ordinary_pixels": rec["pixels"][-8:]})
    assert len(records) == 10 and sum(len(rec["case_pixels"]) for rec in records) == 22
    return records


def candidate_geometry(height, row):
    # Default C95 geometry used by C141. No variants or offset search here.
    axis = np.linspace(-128, 128, 256, dtype=np.float32)
    x, y = np.meshgrid(axis, -axis)
    vertices = np.column_stack((x.ravel(), y.ravel(), height.ravel())).astype(np.float32)
    a = np.arange(65536).reshape(256, 256)[:-1, :-1].ravel()
    faces = np.concatenate((np.stack((a, a+257, a+1), 1), np.stack((a, a+256, a+257), 1)))
    receiver = vertices.copy()
    receiver[:, 2] += 1.
    r, c = int(row["tx_row"]), int(row["tx_col"])
    tx = np.array([c-128., 127.-r, float(np.floor(height[min(r+1, 255), c])+3)])
    return vertices, receiver, faces, tx


def write_ply(path, vertices, faces):
    # The C79 binary layout is used so both backends parse identical PLY bytes.
    data = np.empty(len(faces), dtype=[("n", "u1"), ("idx", "<i4", (3,))])
    data["n"] = 3
    data["idx"] = faces
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.astype("<f4").tofile(stream)
        data.tofile(stream)


def direction_indices(vertices):
    # C86 enumeration, using its frozen shared library without its LLVM imports.
    center = vertices.mean(1)
    radius = np.linalg.norm(vertices - center[:, None], axis=-1).max(1) + 1.35 + .003
    length = np.linalg.norm(center, axis=1)
    delta = np.arcsin(np.minimum(.99, radius / length))
    theta = np.arccos(np.clip(center[:, 2] / length, -1, 1))
    uc = np.arctan2(center[:, 1], center[:, 0]) / (2*np.pi)
    du = np.arcsin(np.minimum(.999, np.sin(delta) / np.sin(theta))) / (2*np.pi)
    vl = .5 * (1 - np.cos(np.maximum(theta-delta, 0)))
    vh = .5 * (1 - np.cos(np.minimum(theta+delta, np.pi)))
    boxes = np.ascontiguousarray(np.stack([uc-du-2e-6, uc+du+2e-6, vl-2e-6, vh+2e-6], 1))
    lib = ctypes.CDLL(str(ROOT / "src/libc86_directions.so"))
    pointer = ctypes.POINTER(ctypes.c_uint32)
    lib.directions86.argtypes = [
        np.ctypeslib.ndpointer(dtype="float64", flags="C_CONTIGUOUS"),
        ctypes.c_int, ctypes.c_longlong, ctypes.POINTER(pointer),
        ctypes.POINTER(ctypes.c_longlong),
    ]
    lib.directions86.restype = ctypes.c_int
    lib.directions86_free.argtypes = [pointer]
    result, count = pointer(), ctypes.c_longlong()
    error = lib.directions86(boxes, len(boxes), N, ctypes.byref(result), ctypes.byref(count))
    if error:
        raise RuntimeError(f"Direction enumeration failed: {error}")
    try:
        return np.ctypeslib.as_array(result, shape=(count.value,)).copy()
    finally:
        lib.directions86_free(result)


def lattice(mi, dr, rays):
    ns = mi.Float64(mi.UInt(rays))
    gold = (1 + dr.sqrt(mi.Float64(5.))) / 2
    u = ns / gold
    u -= dr.floor(u)
    return mi.Point2f(u, ns / (N-1))


def trace_selection(rays, directions, donors, tx):
    # Target each donor's centroid and edge midpoints with fixed global IDs.
    # Add a uniform-in-index sample as a coverage check, without using labels.
    positions = [np.linspace(0, len(rays)-1, 128, dtype=np.int64)]
    targets = np.concatenate((donors.mean(1)[:, None], (donors + donors[:, [1, 2, 0]]) / 2), axis=1)
    for target in targets.reshape(-1, 3):
        toward = target - tx
        toward /= np.linalg.norm(toward)
        center = np.searchsorted(rays, int((1-toward[2]) * .5 * (N-1)))
        lo, hi = max(0, center-8192), min(len(rays), center+8192)
        d = directions[lo:hi]
        dot = d[:, 0] * toward[0] + d[:, 1] * toward[1] + d[:, 2] * toward[2]
        best = np.argsort(-dot, kind="stable")[:4] if len(dot) < 4 else np.argpartition(-dot, 3)[:4]
        positions.append(lo + best)
    return np.unique(np.concatenate(positions))


def freeze(mi, dr, records, maps):
    data = ROOT.parent / "data"
    with (data / "train_index.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    height = np.load(data / "train_hm.npy", mmap_mode="r")
    for rec in records:
        mid = rec["map"]
        if maps and mid not in maps:
            continue
        dest = INPUTS / str(mid)
        if dest.exists():
            continue
        start = time.monotonic()
        work = INPUTS / f"{mid}.partial"
        work.mkdir()
        try:
            row = rows[mid]
            ground, rx, faces, tx = candidate_geometry(height[int(row["hm_row"])], row)
            source = C141 / f"fit_replay/{mid}.npz"
            with np.load(source) as old:
                assert np.array_equal(tx, old["tx"])
                ids, weights, pixels = old["ids"], old["weights"], old["pixels"]
                unique = old["unique_faces"]
                vertices = rx[faces[unique]].astype(float)
                assert np.array_equal(vertices, old["native_vertices"])
                rays = direction_indices(vertices - tx)
                assert len(rays) == int(old["selected_rays"])
            assert rays.dtype == np.uint32 and np.all(rays[1:] > rays[:-1])
            write_ply(work / "ground.ply", ground, faces)
            write_ply(work / "rx.ply", rx, faces)
            np.save(work / "ids.npy", rays, allow_pickle=False)
            d = np.lib.format.open_memmap(work / "directions.npy", mode="w+", dtype=np.float32, shape=(len(rays), 3))
            for offset in range(0, len(rays), BATCH):
                batch = rays[offset:offset+BATCH]
                d[offset:offset+len(batch)] = mi.warp.square_to_uniform_sphere(lattice(mi, dr, batch)).numpy().T
            d.flush()
            use = np.isin(pixels, rec["case_pixels"] + rec["ordinary_pixels"])
            donors = np.unique(ids[use][weights[use] > 1e-12])
            trace_pos = trace_selection(rays, d, rx[faces[donors]].astype(float), tx)
            np.savez_compressed(work / "queries.npz", pixels=pixels, ids=ids, weights=weights, tx=tx, trace_pos=trace_pos)
            del d
            paths = ["ground.ply", "rx.ply", "ids.npy", "directions.npy", "queries.npz"]
            write_json(work / "manifest.json", {
                "map": mid, "rays": len(rays), "trace_rays": len(trace_pos),
                "reference_npz_sha256": file_hash(source),
                "sha256": {name: file_hash(work / name) for name in paths},
            })
            work.replace(dest)
            print(json.dumps({"event": "frozen", "map": mid, "rays": len(rays),
                              "trace_rays": len(trace_pos), "seconds": time.monotonic()-start}), flush=True)
        except BaseException:
            shutil.rmtree(work)
            raise


def load_scene(mi, rt, path, tx):
    ground = mi.load_dict({"type": "ply", "filename": str(path / "ground.ply"),
                          "face_normals": False, "bsdf": {"type": "holder-material"}})
    material = rt.RadioMaterial("regolith", relative_permittivity=3.411,
                               conductivity=.0003562, thickness=10.,
                               scattering_coefficient=.632, xpd_coefficient=.1,
                               scattering_pattern="lambertian")
    scene = rt.load_scene()
    scene.edit(add=rt.SceneObject(mi_mesh=ground, name="terrain", radio_material=material))
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=.5,
                                   horizontal_spacing=.5, pattern="iso", polarization="V")
    scene.rx_array = scene.tx_array
    scene.add(rt.Transmitter("tx", position=tx.tolist()))
    scene.frequency = FREQ
    rx = mi.load_dict({"type": "ply", "filename": str(path / "rx.ply"), "face_normals": True})
    return scene, rx


def trace(mi, dr, scene, rx, tx, directions):
    from sionna.rt.scene import extend_scene_with_mesh

    joint = extend_scene_with_mesh(scene.mi_scene, rx)
    ray = mi.Ray3f(mi.Point3f(*tx.tolist()), mi.Vector3f(directions.T.copy()))
    assert np.array_equal(ray.d.numpy().T, directions)
    active = np.ones(len(directions), dtype=bool)
    kinds, faces, positions, origins, outgoing, cosines = [], [], [], [], [], []
    for _ in range(128):
        si = joint.ray_intersect(ray, active=mi.Bool(active))
        valid = si.is_valid().numpy() & active
        receiver = (si.shape == rx).numpy() & valid
        new_ray = si.spawn_ray(d=ray.d)
        kinds.append(np.where(receiver, 2, np.where(valid, 3, active)).astype(np.uint8))
        faces.append(np.where(valid, si.prim_index.numpy().astype(np.int32), -1))
        positions.append(np.where(valid[:, None], si.p.numpy().T, np.nan))
        origins.append(np.where(receiver[:, None], new_ray.o.numpy().T, np.nan))
        outgoing.append(np.where(receiver[:, None], new_ray.d.numpy().T, np.nan))
        cosines.append(np.where(receiver, dr.abs(dr.dot(si.n, ray.d)).numpy(), np.nan))
        active = receiver
        if not np.any(active):
            break
        ray = new_ray
    if np.any(active):
        raise RuntimeError("Fixed trajectory exceeded 128 receiver crossings")
    return dict(trace_kind=kinds, trace_face=faces, trace_p=positions, trace_o=origins,
                trace_d=outgoing, trace_cos=cosines)


def solve(mi, dr, rt, scene, rx, rays, directions, check_native=False):
    module = importlib.import_module("sionna.rt.radio_map_solvers.radio_map_solver")
    original_spawn, original_lattice = module.spawn_ray_from_sources, module.fibonacci_lattice
    power = np.zeros(rx.face_count(), dtype=np.float64)
    solver = rt.RadioMapSolver()
    native_different = 0
    native_abs_max = 0.
    try:
        for offset in range(0, len(rays), BATCH):
            batch = rays[offset:offset+BATCH]
            batch_d = directions[offset:offset+len(batch)]

            def spawn(_lattice, size, sources):
                nonlocal native_different, native_abs_max
                assert size == len(batch) and dr.shape(sources)[1] == 1
                if check_native:
                    own_d = original_spawn(lambda n: lattice(mi, dr, batch), size, sources).d.numpy().T
                    native_different += int(np.any(own_d != batch_d, axis=1).sum())
                    native_abs_max = max(native_abs_max, float(np.max(np.abs(own_d-batch_d))))
                direction = mi.Vector3f(batch_d.T.copy())
                ray = mi.Ray3f(o=dr.repeat(sources, size), d=direction)
                ray.time = dr.zeros(mi.Float, size)
                return ray

            module.spawn_ray_from_sources = spawn
            result = solver(scene, measurement_surface=rx, samples_per_tx=len(batch), max_depth=0,
                            los=True, specular_reflection=False, diffuse_reflection=False,
                            refraction=False, seed=13)
            power += result.path_gain.numpy()[0].astype(float) * (len(batch)/N) * POWER_SCALE
    finally:
        module.spawn_ray_from_sources = original_spawn
        module.fibonacci_lattice = original_lattice
    assert np.isfinite(power).all() and np.all(power >= 0)
    return power, native_different, native_abs_max


def replay(mi, dr, rt, device, records, maps, backend, score_power):
    out = AUDIT / backend
    out.mkdir(exist_ok=True)
    for rec in records:
        mid = rec["map"]
        if maps and mid not in maps:
            continue
        dest = out / f"{mid}.npz"
        if dest.exists():
            with np.load(dest) as prior:
                if not score_power or "native_full_native" in prior:
                    continue
        start = time.monotonic()
        path = INPUTS / str(mid)
        manifest = json.loads((path / "manifest.json").read_text())
        for name, expected in manifest["sha256"].items():
            assert file_hash(path / name) == expected, (mid, name)
        rays = np.load(path / "ids.npy", mmap_mode="r")
        directions = np.load(path / "directions.npy", mmap_mode="r")
        with np.load(path / "queries.npz") as file:
            queries = {key: file[key] for key in file.files}
        scene, rx = load_scene(mi, rt, path, queries["tx"])
        sel = queries["trace_pos"]
        result = {key: queries[key] for key in ("pixels", "ids", "weights")}
        result.update(trace(mi, dr, scene, rx, queries["tx"], directions[sel]))
        result.update(trace_ids=rays[sel], directions=directions[sel], variant=mi.variant(),
                      gpu_uuid=device or "", input_manifest_sha256=file_hash(path / "manifest.json"),
                      selected_rays=len(rays))
        if score_power:
            power, different, abs_max = solve(mi, dr, rt, scene, rx, rays, directions, check_native=True)
            result.update(native_full_native=power[queries["ids"]], native_direction_differences=different,
                          native_direction_max_abs=abs_max)
        tmp = dest.with_suffix(".partial.npz")
        np.savez_compressed(tmp, **result)
        tmp.replace(dest)
        print(json.dumps({"event": "replayed", "map": mid, "variant": mi.variant(),
                          "gpu_uuid": device, "power": score_power,
                          "seconds": time.monotonic()-start}), flush=True)
        del scene, rx
        dr.flush_malloc_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "trace", "replay"])
    parser.add_argument("--backend", choices=["llvm", "cuda"], default="llvm")
    parser.add_argument("--maps", type=int, nargs="+")
    args = parser.parse_args()
    records = fixed_panel()
    if args.maps and not set(args.maps) <= {rec["map"] for rec in records}:
        parser.error("Use only maps in the frozen C142 panel")
    if args.stage == "freeze" and args.backend != "llvm":
        parser.error("Directions are frozen with LLVM")
    mi, dr, rt, device = rt_environment(args.backend)
    if args.stage == "freeze":
        INPUTS.mkdir(exist_ok=True)
        path = INPUTS / "manifest.json"
        manifest = {
            "records": records, "total_global_rays": N, "batch": BATCH, "seed": 13,
            "versions": [rt.__version__, mi.__version__, dr.__version__],
            "direction_variant": mi.variant(),
            "direction_enumerator_sha256": file_hash(ROOT / "src/libc86_directions.so"),
            "solver_sha256": file_hash(ROOT / ".rt_packages/sionna/rt/radio_map_solvers/radio_map_solver.py"),
        }
        if path.exists():
            assert json.loads(path.read_text()) == manifest
        else:
            write_json(path, manifest)
        freeze(mi, dr, records, args.maps)
    else:
        replay(mi, dr, rt, device, records, args.maps, args.backend, args.stage == "replay")


if __name__ == "__main__":
    main()
