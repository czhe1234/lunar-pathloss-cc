"""Independent continuous LOS bound using interval boxes, without ray tracing or hulls."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".json")
INTERVALS = 8192
SOURCE_MAP = 1819
RX_CELLS = ((22, 38), (23, 37), (23, 38))


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def covered_vertex_range(axis, low, high):
    # Include all vertices of every cell touched by the interval, enlarged
    # slightly to make Float64 boundary comparisons conservative.
    first = max(0, int(np.searchsorted(axis, low-1e-10, side="right"))-1)
    last = min(len(axis)-1, int(np.searchsorted(axis, high+1e-10, side="left")))
    return first, last


def main():
    with (ROOT.parent/"data/train_index.csv").open() as stream:
        row = list(csv.DictReader(stream))[SOURCE_MAP]
    assert row["sample_id"] == "00414_00" and int(row["hm_row"]) == 369
    height = np.load(ROOT.parent/"data/train_hm.npy", mmap_mode="r")[369]
    axis = np.linspace(-128, 128, 256, dtype=np.float32).astype(np.float64)

    # The union of nearest-pixel and floor-pixel conventions is covered by
    # these fractional-index limits. Legacy XY=(99,58) is also included.
    xmin = min(99., np.interp(226.5, np.arange(256), axis))
    xmax = axis[228]
    ymin = -axis[70]
    ymax = -np.interp(68.5, np.arange(256), axis)
    c0, c1 = covered_vertex_range(axis, xmin, xmax)
    r0, r1 = covered_vertex_range(axis, -ymax, -ymin)
    touched = height[r0:r1+1, c0:c1+1].astype(np.float64)
    zmin = min(7., float(np.floor(touched.min())+3.))
    zmax = max(7., float(touched.max()+3.))
    source_vertices = np.array([[x,y,z] for x in (xmin,xmax)
                                for y in (ymin,ymax) for z in (zmin,zmax)])
    intervals = np.arange(INTERVALS+1, dtype=np.float64)/INTERVALS
    lo, hi = intervals[:-1], intervals[1:]
    results = []
    for row_id, col_id in RX_CELLS:
        rc = np.array([[row_id,col_id], [row_id,col_id+1],
                       [row_id+1,col_id+1], [row_id+1,col_id]])
        receiver = np.column_stack((axis[rc[:,1]], -axis[rc[:,0]],
                                    (height[rc[:,0],rc[:,1]]+np.float32(1)).astype(np.float64)))
        rxmin, rxmax = receiver.min(0), receiver.max(0)
        xlow = np.minimum((1-lo)*xmin+lo*rxmin[0], (1-hi)*xmin+hi*rxmin[0])
        xhigh = np.maximum((1-lo)*xmax+lo*rxmax[0], (1-hi)*xmax+hi*rxmax[0])
        ylow = np.minimum((1-lo)*ymin+lo*rxmin[1], (1-hi)*ymin+hi*rxmin[1])
        yhigh = np.maximum((1-lo)*ymax+lo*rxmax[1], (1-hi)*ymax+hi*rxmax[1])
        ray_floor = np.minimum((1-lo)*zmin+lo*rxmin[2], (1-hi)*zmin+hi*rxmin[2])
        ground_ceiling = np.empty(INTERVALS)
        vertex_ranges = []
        for i in range(INTERVALS):
            ca, cb = covered_vertex_range(axis, xlow[i], xhigh[i])
            ra, rb = covered_vertex_range(axis, -yhigh[i], -ylow[i])
            ground_ceiling[i] = np.max(height[ra:rb+1,ca:cb+1])
            vertex_ranges.append([ra,rb,ca,cb])
        gap = ray_floor-ground_ceiling
        worst = int(np.argmin(gap))
        r2 = np.square(source_vertices[:,None,:]-receiver[None,:,:]).sum(2).max()
        results.append({
            "receiver_cell_row_col": [row_id,col_id],
            "receiver_corners": receiver.tolist(),
            "all_intervals_certified_clear": bool(np.all(gap > 1e-10)),
            "nonpositive_or_tolerance_intervals": int(np.count_nonzero(gap <= 1e-10)),
            "minimum_conservative_gap_m": float(gap[worst]),
            "worst_interval": {
                "t": [float(lo[worst]),float(hi[worst])],
                "world_xy_rectangle": [float(xlow[worst]),float(xhigh[worst]),
                                       float(ylow[worst]),float(yhigh[worst])],
                "candidate_vertex_range_rows_cols": vertex_ranges[worst],
                "minimum_ray_z": float(ray_floor[worst]),
                "maximum_ground_vertex_z": float(ground_ceiling[worst]),
            },
            "finite_source_box_max_squared_distance_m2": float(r2),
            "geometric_area_direct_normalized_lower_bound_if_clear": float(1/r2) if np.all(gap>1e-10) else None,
        })
    result = {
        "method": "Interval enclosure of all continuous TX-to-RX segments; no sampled-ray acceptance and no convex-hull algorithm",
        "scope": "Candidate float32 heightfield; declared TX rectangle and receiver cell regions of 00414_00",
        "source_xy_rectangle": [float(xmin),float(xmax),float(ymin),float(ymax)],
        "source_height_lower_bound": zmin,
        "source_height_finite_upper_bound_for_distance_only": zmax,
        "source_touched_vertex_range_rows_cols": [r0,r1,c0,c1],
        "intervals_per_receiver_region": INTERVALS,
        "results": results,
        "all_regions_certified": all(r["all_intervals_certified_clear"] for r in results),
        "proof": "For each closed t interval, endpoint-linear extrema enclose all possible segment XY positions. Every triangular or bilinear candidate height within each touched cell is no larger than its maximum corner height. The segment z is at least the reported endpoint-linear floor. A positive gap therefore excludes every ground intersection throughout every interval.",
        "visibility_monotonicity": "Any source z above zmin raises every interior segment point for fixed XY and RX, so the visibility result, if positive, extends to all z>=zmin. Finite inverse-square bounds use the separately declared finite zmax.",
        "limits": "The TX footprint is an explicit reconstruction uncertainty set, not a recovered original TX. No claim about arbitrary original meshes or native numerical receiver stepping.",
        "labels_or_masks_used": False,
        "gpu_used": False,
        "height_sha256": hashlib.sha256(height.tobytes()).hexdigest(),
        "index_sha256": file_hash(ROOT.parent/"data/train_index.csv"),
    }
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("source_xy_rectangle","source_height_lower_bound",
          "source_height_finite_upper_bound_for_distance_only","all_regions_certified")}))
    print([(r["receiver_cell_row_col"],r["minimum_conservative_gap_m"],
            r["nonpositive_or_tolerance_intervals"]) for r in results])


def cross_method_check():
    from scipy.spatial import ConvexHull

    result = json.loads(OUT.read_text())
    main_path = OUT.with_name("result.json")
    main = json.loads(main_path.read_text())
    geometry_path = OUT.with_name("certificate_geometry.npz")
    with np.load(geometry_path) as geometry:
        assert np.array_equal(geometry["rx_cells"], RX_CELLS)
        assert np.array_equal(
            geometry["source_xy"].min(0),
            [result["source_xy_rectangle"][0], result["source_xy_rectangle"][2]])
        assert np.array_equal(
            geometry["source_xy"].max(0),
            [result["source_xy_rectangle"][1], result["source_xy_rectangle"][3]])
        assert float(geometry["source_zmin"]) == result["source_height_lower_bound"]
        assert float(geometry["source_zmax"]) == result["source_height_finite_upper_bound_for_distance_only"]
        for rec, corners in zip(result["results"], geometry["rx_region_corners"]):
            assert np.array_equal(rec["receiver_corners"], corners)

        facet_checks = []
        for record in main["certificates"]:
            label = record["name"]
            points = geometry[label+"_hull_points"]
            lower = geometry[label+"_lower_facets"]
            hull = ConvexHull(points)
            projection = ConvexHull(points[:, :2])
            triangles = points[lower, :2]
            edges1 = triangles[:,1]-triangles[:,0]
            edges2 = triangles[:,2]-triangles[:,0]
            areas = np.abs(edges1[:,0]*edges2[:,1]-edges1[:,1]*edges2[:,0])/2
            area_gap = float(areas.sum()-projection.volume)
            assert abs(area_gap) < 1e-8
            saved_lower = hull.equations[:,2] < -1e-12
            assert np.array_equal(hull.simplices[saved_lower], lower)
            omitted_down = (hull.equations[:,2] < 0)&~saved_lower
            assert not np.any(omitted_down)
            witness = record["minimum_witness"]
            xy = np.asarray(witness["xy_m"], float)
            ground = np.asarray(witness["ground_triangle_world"], float)
            facet = np.asarray(witness["hull_triangle_world"], float)
            barycentrics = []
            for triangle in (ground, facet):
                uv = np.linalg.solve((triangle[1:,:2]-triangle[0,:2]).T, xy-triangle[0,:2])
                weights = np.r_[1-uv.sum(), uv]
                assert weights.min() >= -1e-10 and weights.max() <= 1+1e-10
                assert np.max(np.abs(weights@triangle[:,:2]-xy)) < 1e-10
                barycentrics.append(weights)
            terrain_z = float(barycentrics[0]@ground[:,2])
            hull_z = float(barycentrics[1]@facet[:,2])
            witness_gap = hull_z-terrain_z
            assert abs(terrain_z-witness["terrain_z_m"]) < 1e-11
            assert abs(hull_z-witness["lower_hull_z_m"]) < 1e-11
            assert abs(witness_gap-record["minimum_gap_m"]) < 1e-11
            facet_checks.append({
                "name": label,
                "lower_projection_area_m2": float(areas.sum()),
                "whole_projection_area_m2": float(projection.volume),
                "area_difference_m2": area_gap,
                "minimum_lower_facet_projected_area_m2": float(areas.min()),
                "omitted_downward_facets": int(omitted_down.sum()),
                "witness_gap_independently_recomputed_m": witness_gap,
                "witness_ground_barycentric": barycentrics[0].tolist(),
                "witness_hull_barycentric": barycentrics[1].tolist(),
            })

    max_r2 = max(rec["finite_source_box_max_squared_distance_m2"]
                 for rec in result["results"])
    frequencies = np.array([415e6,5.775e9])
    pl_upper = 10*np.log10(max_r2)+20*np.log10(4*np.pi*frequencies/299792458.)
    labels = np.array([np.load(ROOT.parent/f"data/train_pl{band}.npy",mmap_mode="r")[1819].ravel()[5926]
                       for band in ("415","58")],float)
    masks = [bool(np.unpackbits(np.load(ROOT.parent/f"data/train_mask{band}.npy",mmap_mode="r")[1819])[5926])
             for band in ("415","58")]
    reference = main["conditional_ideal_area_direct_power"]
    assert all(masks) and np.array_equal(labels, reference["published_label_pathloss_dB"])
    assert abs(np.sqrt(max_r2)-reference["max_endpoint_distance_m"]) < 1e-12
    assert np.max(np.abs(pl_upper-reference["pathloss_upper_bound_dB"])) < 2e-14
    result["main_cross_method_checks"] = {
        "same_TX_and_RX_domains": True,
        "same_candidate_height_hash": result["height_sha256"] == main["sources"]["published_height_sample_sha256"],
        "lower_hull_projection_covers_full_domain": True,
        "facet_checks": facet_checks,
        "independent_max_endpoint_distance_m": float(np.sqrt(max_r2)),
        "independent_ideal_area_pathloss_upper_bound_dB": pl_upper.tolist(),
        "independent_ideal_direct_excess_above_label_dB": (labels-pl_upper).tolist(),
        "raw_label_and_masks_match": True,
        "main_result_sha256": file_hash(main_path),
        "main_geometry_sha256": file_hash(geometry_path),
    }
    assert result["main_cross_method_checks"]["same_candidate_height_hash"]
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result["main_cross_method_checks"]


if __name__ == "__main__":
    main()
    if OUT.with_name("result.json").exists() and OUT.with_name("certificate_geometry.npz").exists():
        cross_method_check()
