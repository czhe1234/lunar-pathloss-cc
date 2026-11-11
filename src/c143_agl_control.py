"""C143 one-factor exact-AGL control on the fixed 00414_00 C141 queries."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from c142_backend import candidate_geometry, load_scene, rt_environment, write_json

OUT = ROOT / "audit/c143_case00414/agl_control"
C141 = ROOT / "audit/c141_direct_receiver"
MID = 1819
PIXEL = 5926
FREQUENCIES = np.array([415e6, 5.775e9])
WAVELENGTH_CONSTANT_DB = 20 * np.log10(4 * np.pi * FREQUENCIES / 299792458.)


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def quadrature(order):
    barycentric = []
    for i in range(order):
        for j in range(order - i):
            for fraction in ((1/3, 2/3) if i+j+1 < order else (1/3,)):
                u = (i+fraction)/order
                v = (j+fraction)/order
                barycentric.append([1-u-v, u, v])
    return np.array(barycentric, float)


def surface_height(vertices, faces, x, y):
    # Candidate C95 main diagonal at the fixed world XY.
    u = (x+128) * 255/256
    v = (128-y) * 255/256
    col = min(254, int(u))
    row = min(254, int(v))
    u -= col
    v -= row
    cell = row*255+col
    if u >= v:
        face_id = cell
        weights = np.array([1-u, v, u-v])
    else:
        face_id = 255*255+cell
        weights = np.array([1-v, v-u, u])
    position = (vertices[faces[face_id]].astype(float) * weights[:, None]).sum(0)
    assert np.max(np.abs(position[:2]-[x,y])) < 2e-5
    return position[2], face_id, weights


def pathloss(power):
    with np.errstate(divide="ignore"):
        return -10*np.log10(power)


def power_summary(before, after, weights, labels, selected):
    positive = weights > 1e-12
    old_pl = pathloss(before)
    new_pl = pathloss(after)
    old_average = (before * weights).sum(1)
    new_average = (after * weights).sum(1)
    old_weakest = np.max(np.where(positive, old_pl, -np.inf), axis=1)[:, None] + WAVELENGTH_CONSTANT_DB
    new_weakest = np.max(np.where(positive, new_pl, -np.inf), axis=1)[:, None] + WAVELENGTH_CONSTANT_DB
    shift = pathloss(new_average)-pathloss(old_average)
    return {
        "points": int(selected.sum()),
        "max_abs_direct_shift_db": float(np.max(np.abs(shift[selected]))),
        "old_weakest_donor_violations_over_1dB_per_band":
            np.sum(selected[:, None] & (labels-old_weakest > 1), axis=0).tolist(),
        "corrected_weakest_donor_violations_over_1dB_per_band":
            np.sum(selected[:, None] & (labels-new_weakest > 1), axis=0).tolist(),
        "old_weakest_finite_points_per_band":
            np.sum(selected[:, None] & np.isfinite(old_weakest), axis=0).tolist(),
        "corrected_weakest_finite_points_per_band":
            np.sum(selected[:, None] & np.isfinite(new_weakest), axis=0).tolist(),
    }


def metric(prediction, labels, selection):
    n = int(selection.sum())
    error = prediction[selection]-labels[selection]
    assert np.isfinite(error).all()
    return {"n": n, "RMSE_db": np.sqrt(np.mean(error**2, axis=0)).tolist()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "area256.npz"
    summary_path = OUT / "result.json"
    if destination.exists() or summary_path.exists():
        raise SystemExit("C143 AGL artifacts already exist; reuse the saved result")
    with (ROOT.parent / "data/train_index.csv").open() as file:
        rows = list(csv.DictReader(file))
    row = rows[MID]
    assert row["sample_id"] == "00414_00"
    height = np.load(ROOT.parent / "data/train_hm.npy", mmap_mode="r")[int(row["hm_row"])]
    vertices, receiver_vertices, faces, old_tx = candidate_geometry(height, row)
    input_path = ROOT / "audit/c142_backend/inputs/1819"
    manifest = json.loads((input_path/"manifest.json").read_text())
    for name in ("ground.ply", "rx.ply"):
        assert sha256(input_path/name) == manifest["sha256"][name]
    metadata = json.loads((ROOT.parent/"data/metadata.json").read_text())
    assert metadata["simulation"]["tx_height_m_agl"] == 3.
    with np.load(C141 / "fit_replay/1819.npz") as cache:
        pixels = cache["pixels"]
        ids = cache["ids"]
        weights = cache["weights"]
        unique_faces = cache["unique_faces"]
        assert np.array_equal(cache["tx"], old_tx)
        assert np.array_equal(receiver_vertices[faces[unique_faces]].astype(float), cache["native_vertices"])
        old_area = cache["area256_native"]
        old_area4 = cache["area4_native"]
        old_los = cache["area256_los"]
    z, tx_face, tx_weights = surface_height(vertices, faces, old_tx[0], old_tx[1])
    corrected_tx = old_tx.copy()
    corrected_tx[2] = z+3.
    mi, dr, rt, device = rt_environment("llvm")
    scene, receiver = load_scene(mi, rt, input_path, corrected_tx)
    ray = mi.Ray3f(mi.Point3f(float(old_tx[0]), float(old_tx[1]), 1000), mi.Vector3f(0,0,-1))
    si = scene.mi_scene.ray_intersect(ray)
    assert si.is_valid()[0]
    assert abs(float(si.p.z[0])-z) < 2e-5
    query_vertices = receiver_vertices[faces[unique_faces]].astype(float)
    points = np.einsum("qv,fvc->fqc", quadrature(16), query_vertices).reshape(-1,3)
    delta = points-corrected_tx
    distance = np.linalg.norm(delta,axis=1)
    direction = delta/distance[:,None]
    rays = mi.Ray3f(mi.Point3f(*corrected_tx.tolist()),mi.Vector3f(direction.T.astype(np.float32)))
    rays.maxt = mi.Float((distance-.0008).astype(np.float32))
    visible = ~scene.mi_scene.ray_test(rays).numpy()
    unique_power = (visible/np.maximum(distance**2,.0625)).reshape(len(unique_faces),256).mean(1)
    unique_visibility = visible.reshape(len(unique_faces),256).mean(1)
    lookup = {int(face): index for index, face in enumerate(unique_faces)}
    inverse = np.array([[lookup[int(face)] for face in line] for line in ids])
    corrected_area = unique_power[inverse]
    corrected_los = (unique_visibility[inverse]*weights).sum(1)
    assert np.isfinite(corrected_area).all() and np.all(corrected_area >= 0)
    # Save the label-free geometry output before scoring.
    np.savez_compressed(
        destination,pixels=pixels,ids=ids,weights=weights,
        old_tx=old_tx,corrected_tx=corrected_tx,
        area256_native=corrected_area,area256_los=corrected_los,
        original_area256_native=old_area,original_area256_los=old_los,
        surface_height=z,tx_face=tx_face,tx_face_weights=tx_weights,
    )
    scoring_record = next(rec for rec in json.loads((C141/"fit_panel.json").read_text())["records"] if rec["map"]==MID)
    assert scoring_record["pixels"] == pixels.tolist()
    labels = np.asarray(scoring_record["labels"],float)
    c82 = np.asarray(scoring_record["C82"],float)
    cases = np.asarray(scoring_record["is_case"],bool)
    with np.load(ROOT/"audit/c136_native/cache/1819.npz") as cache:
        inverse = cache["inv"][scoring_record["cache_positions"]]
        assert np.array_equal(cache["uf"][inverse],ids)
        dw = cache["dw"][inverse].astype(float)
    background = np.maximum(np.divide(
        old_area4[...,None]*(1-dw),dw,
        out=np.zeros_like(dw),where=dw>0
    ),0)
    def total_prediction(area):
        return pathloss(((area[...,None]+background)*weights[...,None]).sum(1))+WAVELENGTH_CONSTANT_DB
    old_prediction = total_prediction(old_area)
    corrected_prediction = total_prediction(corrected_area)
    j = int(np.flatnonzero(pixels == PIXEL)[0])
    positive = weights[j]>1e-12
    new_weakest = pathloss(corrected_area[j,positive]).max()+WAVELENGTH_CONSTANT_DB
    old_weakest = pathloss(old_area[j,positive]).max()+WAVELENGTH_CONSTANT_DB
    old_direct = pathloss((old_area[j]*weights[j]).sum())+WAVELENGTH_CONSTANT_DB
    new_direct = pathloss((corrected_area[j]*weights[j]).sum())+WAVELENGTH_CONSTANT_DB
    result = {
        "scope": "Fixed 00414_00 C141 TRAIN queries; 8 enriched cases and 8 ordinary controls",
        "frequency_hz":FREQUENCIES.tolist(),
        "published_metadata_tx_height_m_agl":metadata["simulation"]["tx_height_m_agl"],
        "published_metadata_sha256":sha256(ROOT.parent/"data/metadata.json"),
        "backend":mi.variant(),"gpu_used":device is not None,
        "cpu_threads":2,
        "assumption":"Candidate ground/RX meshes and world XY held fixed. Only TX height changed to exactly 3 m above this candidate surface.",
        "actual_original_mesh_or_TX_recovered":False,
        "published_tx_pixel":[int(row["tx_row"]),int(row["tx_col"])],
        "hm_row":int(row["hm_row"]),
        "old_tx":old_tx.tolist(),"candidate_surface_height_at_tx_world_xy":float(z),
        "candidate_surface_ray_height_check":float(si.p.z[0]),
        "old_tx_actual_agl":float(old_tx[2]-z),
        "corrected_tx":corrected_tx.tolist(),
        "corrected_tx_actual_agl":float(corrected_tx[2]-z),
        "ground_ply_sha256":sha256(input_path/"ground.ply"),
        "receiver_ply_sha256":sha256(input_path/"rx.ply"),
        "baseline_npz_sha256":sha256(C141/"fit_replay/1819.npz"),
        "quadrature_points_per_donor":256,
        "positive_weighted_query_visibility_changes":int(np.count_nonzero(old_los != corrected_los)),
        "old_query_visibility":old_los.tolist(),
        "corrected_query_visibility":corrected_los.tolist(),
        "target":{
            "pixel":PIXEL,"row":23,"col":38,
            "label_db":labels[j].tolist(),"C82_db":c82[j].tolist(),
            "old_area256_direct_db":old_direct.tolist(),
            "corrected_area256_direct_db":new_direct.tolist(),
            "old_area256_weakest_direct_db":old_weakest.tolist(),
            "corrected_area256_weakest_direct_db":new_weakest.tolist(),
            "old_area256_weakest_violation_db":(labels[j]-old_weakest).tolist(),
            "corrected_area256_weakest_violation_db":(labels[j]-new_weakest).tolist(),
            "donor_ids":ids[j].tolist(),"weights":weights[j].tolist(),
        },
        "groups": {},
        "native_full_replay_performed":False,
        "native_full_replay_reason":"No area256 visibility changes on the fixed case or ordinary donors; direct-area sensitivity is reported before considering further receiver replay.",
        "interpretation":"This exact-AGL control tests a reconstruction/metadata discrepancy. Its result does not establish the actual original TX.",
    }
    for name,selected in (("all",np.ones(len(pixels),bool)),("cases",cases),("ordinary",~cases)):
        result["groups"][name]={
            **power_summary(old_area,corrected_area,weights,labels,selected),
            "total_with_C141_fixed_background_old_area256":metric(old_prediction,labels,selected),
            "total_with_C141_fixed_background_corrected_area256":metric(corrected_prediction,labels,selected),
            "C82":metric(c82,labels,selected),
        }
    # No visibility change means the source-motivated TX adjustment leaves this
    # independent area direct-power explanation intact. Native stepping is a separate mechanism.
    if result["positive_weighted_query_visibility_changes"]:
        result["native_full_replay_reason"]="Visibility changed; a native corrected-TX replay is warranted."
    write_json(summary_path,result)
    print(json.dumps({"target":result["target"],"groups":result["groups"],
                      "visibility_changes":result["positive_weighted_query_visibility_changes"]},indent=2))
    dr.flush_malloc_cache()


if __name__=="__main__":
    main()
