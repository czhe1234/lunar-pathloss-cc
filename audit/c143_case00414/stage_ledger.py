"""Cache-only stage ledger for TRAIN 00414_00, pixel (23,38).
Original generating mesh/TX/cell arrays are unavailable in the checked release.
This script documents reconstructed stages without claiming original equivalence.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from c142_backend import candidate_geometry

OUT = Path(__file__).resolve().parent
C141 = ROOT/"audit/c141_direct_receiver"
C142 = ROOT/"audit/c142_backend"
DATA = ROOT.parent/"data"
MID = 1819
PIXEL = 5926
FREQUENCIES = np.array([415e6, 5.775e9])
GAIN_FACTOR = (299792458./(4*np.pi*FREQUENCIES))**2
PL_CONSTANT = -10*np.log10(GAIN_FACTOR)


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def pl(power):
    with np.errstate(divide="ignore"):
        return -10*np.log10(power)


def exported(donors, weights):
    positive = weights > 1e-12
    donor_db = pl(donors)
    finite_log = np.where(positive, donor_db, 0)
    return {
        "power_domain_PL_dB": pl(np.sum(donors*weights, axis=0))+PL_CONSTANT,
        "db_domain_PL_dB": np.sum(finite_log*weights, axis=0)+PL_CONSTANT,
        "weakest_positive_donor_PL_dB": np.max(np.where(positive, donor_db, -np.inf), axis=0)+PL_CONSTANT,
    }


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {key:clean(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def scalar_surface_height(height, ground, tx):
    # Use the exact Float32 vertex coordinates stored in the candidate PLY.
    x_axis=ground[:,0].reshape(256,256)[0].astype(float)
    y_axis=ground[:,1].reshape(256,256)[:,0].astype(float)
    col=int(np.searchsorted(x_axis,tx[0],side="right")-1)
    row=int(np.searchsorted(-y_axis,-tx[1],side="right")-1)
    col=min(254,max(0,col))
    row=min(254,max(0,row))
    u=(tx[0]-x_axis[col])/(x_axis[col+1]-x_axis[col])
    v=(y_axis[row]-tx[1])/(y_axis[row]-y_axis[row+1])
    if u>=v:
        return float((1-u)*height[row,col]+(u-v)*height[row,col+1]+v*height[row+1,col+1])
    return float((1-v)*height[row,col]+u*height[row+1,col+1]+(v-u)*height[row+1,col])


def main():
    source_paths = [
        ROOT/"src/c142_backend.py", ROOT/"src/c141_direct_replay.py",
        C141/"fit_panel.json", C141/"fit_replay/1819.npz",
        C142/"inputs/1819/manifest.json", C142/"inputs/1819/queries.npz",
        C142/"inputs/1819/ground.ply", C142/"inputs/1819/rx.ply",
        C142/"cuda/1819.npz", ROOT/"audit/c136_native/cache/1819.npz",
        DATA/"metadata.json", DATA/"README.md", DATA/"train_index.csv",
    ]
    panel = json.loads((C141/"fit_panel.json").read_text())["records"]
    record = next(row for row in panel if row["map"]==MID)
    assert record["sample_id"]=="00414_00"
    rows=list(csv.DictReader((DATA/"train_index.csv").open()))
    raw_row=rows[MID]
    height=np.load(DATA/"train_hm.npy",mmap_mode="r")[int(raw_row["hm_row"])]
    ground,rx,faces,tx = candidate_geometry(height,raw_row)
    old=np.load(C141/"fit_replay/1819.npz")
    cuda=np.load(C142/"cuda/1819.npz")
    query=np.load(C142/"inputs/1819/queries.npz")
    assert np.array_equal(tx,old["tx"])
    for key in ("pixels","ids","weights"):
        assert np.array_equal(old[key],cuda[key]) and np.array_equal(old[key],query[key])
    for face,vertices in zip(old["unique_faces"],old["native_vertices"]):
        assert np.array_equal(rx[faces[face]].astype(float),vertices)

    pixels=old["pixels"]
    labels=np.column_stack([
        np.load(DATA/f"train_pl{band}.npy",mmap_mode="r")[MID].ravel()[pixels]
        for band in ("415","58")
    ]).astype(float)
    for band in ("415","58"):
        valid=np.unpackbits(np.load(DATA/f"train_mask{band}.npy",mmap_mode="r")[MID])
        assert np.all(valid[pixels])
    assert np.array_equal(labels,np.asarray(record["labels"],float))
    cache=np.load(ROOT/"audit/c136_native/cache/1819.npz")
    inverse=cache["inv"][record["cache_positions"]]
    assert np.array_equal(cache["uf"][inverse],old["ids"])
    direct_fraction=cache["dw"][inverse].astype(float)
    base_direct=old["area4_native"][...,None]
    assert np.all((direct_fraction>0)|(old["weights"][...,None]<1e-7))
    extra=np.maximum(np.divide(base_direct*(1-direct_fraction),direct_fraction,
                               out=np.zeros_like(direct_fraction),where=direct_fraction>0),0)

    entries=[]
    for index,pixel in enumerate(pixels):
        weights=old["weights"][index,:,None]
        stages={}
        for name in ("area4","area16","area64","area256","native_full"):
            donor=old[name+"_native"][index,:,None]
            stages[name] = exported(donor,weights)
            stages[name]["normalized_donor_power"]=donor[:,0]
            stages[name]["positive_weight_donors_with_positive_power"]=int(
                np.count_nonzero((weights[:,0]>1e-12)&(donor[:,0]>0)))
        donor=cuda["native_full_native"][index,:,None]
        stages["cuda_native_full"]=exported(donor,weights)
        stages["cuda_native_full"]["normalized_donor_power"]=donor[:,0]
        stages["cuda_native_full"]["positive_weight_donors_with_positive_power"]=int(
            np.count_nonzero((weights[:,0]>1e-12)&(donor[:,0]>0)))
        for name,direct in (("C141",old["native_full_native"][index,:,None]),
                            ("C142_CUDA",donor)):
            stages[name+"_with_saved_background"]=exported(direct+extra[index],weights)
        direct_pl=stages["cuda_native_full"]["power_domain_PL_dB"]
        entry={
            "pixel":int(pixel),"row":int(pixel//256),"col":int(pixel%256),
            "is_enriched_case":bool(record["is_case"][index]),
            "published_label_PL_dB":labels[index],
            "C82_PL_dB":record["C82"][index],
            "donor_ids":old["ids"][index],"weights":old["weights"][index],
            "donor_vertex_indices":faces[old["ids"][index]],
            "donor_vertex_rows":faces[old["ids"][index]]//256,
            "donor_vertex_cols":faces[old["ids"][index]]%256,
            "candidate_receiver_vertices":rx[faces[old["ids"][index]]],
            "candidate_receiver_centroids":rx[faces[old["ids"][index]]].astype(float).mean(1),
            "saved_background_normalized_donor_power":extra[index],
            "stages":stages,
            "label_minus_cuda_mapped_direct_dB":labels[index]-direct_pl,
            "allowed_label_total_over_cuda_mapped_direct_ratio":10**(-(labels[index]-direct_pl)/10),
            "label_minus_cuda_weakest_donor_dB":labels[index]-stages["cuda_native_full"]["weakest_positive_donor_PL_dB"],
            "cuda_minus_C141_mapped_direct_dB":direct_pl-stages["native_full"]["power_domain_PL_dB"],
            "native_receiver_minus_area256_mapped_direct_dB":stages["native_full"]["power_domain_PL_dB"]-stages["area256"]["power_domain_PL_dB"],
        }
        entries.append(entry)
    selected=next(entry for entry in entries if entry["pixel"]==PIXEL)
    ordinary=[entry for entry in entries if not entry["is_enriched_case"]]
    assert len(ordinary)==8
    summaries={}
    for stage in ("C141_with_saved_background","C142_CUDA_with_saved_background"):
        for domain in ("power_domain_PL_dB","db_domain_PL_dB"):
            prediction=np.asarray([entry["stages"][stage][domain] for entry in ordinary])
            truth=np.asarray([entry["published_label_PL_dB"] for entry in ordinary])
            finite=np.isfinite(prediction)
            residual=np.where(finite,prediction-truth,0)
            count=finite.sum(0)
            summaries[f"{stage}_{domain}"]={
                "finite_count":count,
                "RMSE_dB":np.sqrt((residual**2).sum(0)/np.maximum(count,1)),
            }
    c82=np.asarray([entry["C82_PL_dB"] for entry in ordinary])
    truth=np.asarray([entry["published_label_PL_dB"] for entry in ordinary])
    summaries["C82"]={"finite_count":[8,8],"RMSE_dB":np.sqrt(((c82-truth)**2).mean(0))}
    ordinary_physical=np.asarray([entry["stages"]["C142_CUDA_with_saved_background"]["power_domain_PL_dB"] for entry in ordinary])
    case_global_shift=np.asarray(selected["published_label_PL_dB"])-selected["stages"]["C142_CUDA_with_saved_background"]["power_domain_PL_dB"]
    normalization_control={
        "ordinary_mean_label_minus_physical_PL_dB":np.mean(truth-ordinary_physical,axis=0),
        "ordinary_max_absolute_error_dB":np.max(np.abs(truth-ordinary_physical),axis=0),
        "global_shift_needed_to_fit_counterexample_exactly_dB":case_global_shift,
        "ordinary_RMSE_after_that_same_global_shift_dB":np.sqrt(np.mean((ordinary_physical+case_global_shift-truth)**2,axis=0)),
        "interpretation":"One common map-wide dB/unit correction cannot simultaneously fit this counterexample and these ordinary points under the saved candidate model.",
    }
    ground_z=scalar_surface_height(height,ground,tx)
    result={
        "scope":"TRAIN 00414_00, map1819, fixed diagnostic point (23,38) and the same eight C141 ordinary controls.",
        "frequencies_hz":FREQUENCIES,
        "computation":"Existing C141/C136/C142 caches only; no new rays, fitted parameters, labels in geometry or GPU calls.",
        "raw_index_row":raw_row,
        "published_quantities":{"height_shape":height.shape,"height_dtype":str(height.dtype),
            "height_sha256":hashlib.sha256(height.tobytes()).hexdigest(),
            "TX_pixel_row_col":[int(raw_row["tx_row"]),int(raw_row["tx_col"])],
            "labels_checked_against_original_arrays":True,"both_masks_valid_at_all_recorded_points":True},
        "original_quantities_unavailable_for_direct_comparison":[
            "Original generating terrain vertices, topology and absolute coordinate transform",
            "Original continuous Transmitter.position for this exact map",
            "Original measurement mesh vertices and topology",
            "Original pre-griddata path_gain/cell_centers and all seed outputs",
            "Exact per-stage export query coordinates, support filtering and power/dB transforms",
        ],
        "candidate_scene":{
            "provenance":"Default geometry copied from C95 to C142; reconstructed from published height/TX, not original generating geometry.",
            "vertex_axis":"linspace(-128,128,256) and reversed y, Float32",
            "faces":"Main diagonal, upward winding, 130050 faces",
            "TX_world":tx,"TX_z_rule":"floor(H[row+1,col])+3",
            "H_at_published_TX_pixel":float(height[int(raw_row["tx_row"]),int(raw_row["tx_col"])]),
            "H_at_next_row_same_col":float(height[int(raw_row["tx_row"])+1,int(raw_row["tx_col"])]),
            "candidate_ground_z_at_TX_xy":ground_z,"candidate_TX_actual_AGL_m":float(tx[2]-ground_z),
            "TX_z_if_exactly_3m_above_same_candidate_xy":ground_z+3,
            "receiver":"Same candidate terrain vertices shifted +1m along world z",
        },
        "cuda_evidence":{"variant":str(cuda["variant"]),"gpu_uuid":str(cuda["gpu_uuid"]),
            "ray_count":int(cuda["selected_rays"]),"native_launch_direction_differences":int(cuda["native_direction_differences"]),
            "native_launch_direction_max_absolute":float(cuda["native_direction_max_abs"])},
        "selected_counterexample":selected,
        "ordinary_summary":summaries,
        "map_wide_normalization_control":normalization_control,
        "ordinary_controls":ordinary,
        "other_same_map_enriched_cases":[entry for entry in entries if entry["is_enriched_case"] and entry["pixel"]!=PIXEL],
        "stage_location_limit":"Available stages show the mismatch between the reconstructed native receiver-cell powers and the published pixel label. No original intermediate values exist here to identify the first true generating stage.",
        "conditional_TX_height_observation":"At fixed candidate terrain and TX xy, setting TX AGL to nominal 3m raises source z. The straight segments to the already visible receiver samples remain above the same heightfield. This height-only correction cannot create the required attenuation; original xy and mesh are still unknown.",
        "sha256":{str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):sha(path)
                  for path in source_paths},
    }
    out=OUT/"stage_ledger.json"
    out.write_text(json.dumps(clean(result),indent=2,allow_nan=False)+"\n")
    print(json.dumps(clean({
        "case":{key:selected[key] for key in ("published_label_PL_dB","C82_PL_dB",
            "label_minus_cuda_mapped_direct_dB","allowed_label_total_over_cuda_mapped_direct_ratio",
            "cuda_minus_C141_mapped_direct_dB","native_receiver_minus_area256_mapped_direct_dB")},
        "TX_candidate_actual_AGL_m":result["candidate_scene"]["candidate_TX_actual_AGL_m"],
        "ordinary":summaries,
    }),indent=2,allow_nan=False))


if __name__=="__main__":
    main()
