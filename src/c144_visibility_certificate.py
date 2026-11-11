"""C144 continuous LOS certificate for one fixed TRAIN counterexample.

The certificate concerns piecewise planar candidate terrain and ideal geometric
visibility. It does not bound numerical receiver-surface stepping.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from c142_backend import candidate_geometry


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit/c144_visibility"
MID = 1819
PIXEL = 5926
CLIP_TOL = 1e-10
GAP_TOL = 1e-8


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(".partial.json")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def ccw(vertices):
    result = np.asarray(vertices, dtype=float)
    if cross2(result[1] - result[0], result[2] - result[0]) < 0:
        result = result[::-1]
    return result


def clip_polygon(subject, triangle):
    """Clip a convex polygon against one counterclockwise triangle."""
    polygon = [point.copy() for point in subject]
    for index in range(3):
        origin = triangle[index]
        edge = triangle[(index + 1) % 3] - origin
        clipped = []
        if not polygon:
            break
        previous = polygon[-1]
        previous_side = float(cross2(edge, previous-origin))
        for current in polygon:
            current_side = float(cross2(edge, current-origin))
            previous_inside = previous_side >= -CLIP_TOL
            current_inside = current_side >= -CLIP_TOL
            if previous_inside != current_inside:
                denominator = previous_side-current_side
                if abs(denominator) > 1e-30:
                    fraction = np.clip(previous_side/denominator, 0., 1.)
                    clipped.append(previous+fraction*(current-previous))
            if current_inside:
                clipped.append(current)
            previous, previous_side = current, current_side
        polygon = clipped
    return np.asarray(polygon, dtype=float).reshape(-1, 2)


def plane_height(points, triangle):
    delta = triangle[1:]-triangle[0]
    normal = np.cross(delta[0], delta[1])
    assert normal[2] != 0
    return triangle[0, 2] - (
        normal[0] * (points[:, 0]-triangle[0, 0]) +
        normal[1] * (points[:, 1]-triangle[0, 1])
    ) / normal[2]


def barycentric_xy(point, triangle):
    matrix = (triangle[1:, :2]-triangle[0, :2]).T
    weights = np.linalg.solve(matrix, point-triangle[0, :2])
    return np.array([1-weights.sum(), weights[0], weights[1]])


def candidate_faces():
    a = np.arange(65536).reshape(256, 256)[:-1, :-1].ravel()
    main = np.concatenate((
        np.stack((a, a+257, a+1), 1),
        np.stack((a, a+256, a+257), 1),
    )).astype(np.int32)
    anti = np.concatenate((
        np.stack((a, a+256, a+1), 1),
        np.stack((a+257, a+1, a+256), 1),
    )).astype(np.int32)
    return main, anti


def source_domain(vertices, old_tx, row):
    axis = vertices[:256, 0].astype(float)
    index = np.arange(256)
    tx_row, tx_col = int(row["tx_row"]), int(row["tx_col"])
    index_x = [tx_col-.5, tx_col+1.]
    index_y = [tx_row-.5, tx_row+1.]
    xmin = min(old_tx[0], float(np.interp(index_x[0], index, axis)))
    xmax = max(old_tx[0], float(np.interp(index_x[1], index, axis)))
    ymin = min(old_tx[1], -float(np.interp(index_y[1], index, axis)))
    ymax = max(old_tx[1], -float(np.interp(index_y[0], index, axis)))
    xy = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return xy, index_x, index_y


def touched_cells(vertices, xy):
    """Include every grid cell whose closed XY box touches the rectangle."""
    points = vertices.reshape(256, 256, 3)
    xmin, ymin = xy.min(0)
    xmax, ymax = xy.max(0)
    x_axis = points[0, :, 0].astype(float)
    y_axis = points[:, 0, 1].astype(float)
    cols = np.flatnonzero((x_axis[:-1] <= xmax+CLIP_TOL) &
                         (x_axis[1:] >= xmin-CLIP_TOL))
    rows = np.flatnonzero((y_axis[1:] <= ymax+CLIP_TOL) &
                         (y_axis[:-1] >= ymin-CLIP_TOL))
    cells = np.array([(r, c) for r in rows for c in cols], dtype=np.int32)
    corners = []
    for r, c in cells:
        corners.extend((r*256+c, r*256+c+1, (r+1)*256+c, (r+1)*256+c+1))
    return cells, np.unique(corners)


def certificate(vertices, faces, source, receiver, label, saved_geometry):
    hull_points = np.concatenate((source, receiver))
    hull = ConvexHull(hull_points)
    lower = hull.simplices[hull.equations[:, 2] < -1e-12]
    saved_geometry[label+"_hull_points"] = hull_points
    saved_geometry[label+"_lower_facets"] = lower
    ground = vertices[faces].astype(float)
    ground_min = ground[:, :, :2].min(1)
    ground_max = ground[:, :, :2].max(1)
    best = None
    checked_polygons = 0
    checked_vertices = 0
    for facet_index, facet in enumerate(lower):
        hull_triangle = hull_points[facet]
        clip = ccw(hull_triangle[:, :2])
        clip_min, clip_max = clip.min(0), clip.max(0)
        keep = np.all(ground_max >= clip_min-CLIP_TOL, axis=1)
        keep &= np.all(ground_min <= clip_max+CLIP_TOL, axis=1)
        candidates = np.flatnonzero(keep)
        for edge_index in range(3):
            origin = clip[edge_index]
            edge = clip[(edge_index+1) % 3]-origin
            distance = cross2(edge, ground[candidates, :, :2]-origin)
            candidates = candidates[~np.all(distance < -CLIP_TOL, axis=1)]
        for ground_index in candidates:
            triangle = ground[ground_index]
            polygon = clip_polygon(triangle[:, :2], clip)
            if not len(polygon):
                continue
            checked_polygons += 1
            checked_vertices += len(polygon)
            hull_z = plane_height(polygon, hull_triangle)
            terrain_z = plane_height(polygon, triangle)
            gaps = hull_z-terrain_z
            point_index = int(np.argmin(gaps))
            minimum = float(gaps[point_index])
            if best is None or minimum < best["gap_m"]:
                point = polygon[point_index]
                best = {
                    "gap_m": minimum,
                    "xy_m": point.tolist(),
                    "terrain_z_m": float(terrain_z[point_index]),
                    "lower_hull_z_m": float(hull_z[point_index]),
                    "ground_face": int(ground_index),
                    "ground_vertex_ids": faces[ground_index].tolist(),
                    "ground_triangle_world": triangle.tolist(),
                    "hull_facet": int(facet_index),
                    "hull_vertex_ids": facet.tolist(),
                    "hull_triangle_world": hull_triangle.tolist(),
                    "ground_barycentric_xy": barycentric_xy(point, triangle).tolist(),
                    "hull_barycentric_xy": barycentric_xy(point, hull_triangle).tolist(),
                    "intersection_polygon_xy": polygon.tolist(),
                }
    assert best is not None and checked_polygons > 0
    assert min(best["ground_barycentric_xy"]) >= -1e-7
    assert min(best["hull_barycentric_xy"]) >= -1e-7
    return {
        "name": label,
        "lower_facets": len(lower),
        "intersected_ground_triangle_facet_polygons": checked_polygons,
        "intersection_polygon_vertices_evaluated": checked_vertices,
        "minimum_gap_m": best["gap_m"],
        "positive_with_tolerance": best["gap_m"] > GAP_TOL,
        "minimum_witness": best,
    }


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/"result.json").exists() or (OUT/"certificate_geometry.npz").exists():
        raise SystemExit("C144 certificate already saved. Reuse existing results.")
    with (ROOT.parent/"data/train_index.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    row = rows[MID]
    assert row["sample_id"] == "00414_00"
    height = np.load(ROOT.parent/"data/train_hm.npy", mmap_mode="r")[int(row["hm_row"])]
    vertices, receiver_vertices, main_faces, old_tx = candidate_geometry(height, row)
    main, anti = candidate_faces()
    assert np.array_equal(main, main_faces)
    c141_path = ROOT/"audit/c141_direct_receiver/fit_replay/1819.npz"
    with np.load(c141_path) as original:
        assert np.array_equal(old_tx, original["tx"])
        assert np.array_equal(receiver_vertices[main[original["unique_faces"]]].astype(float),
                              original["native_vertices"])
        query = int(np.flatnonzero(original["pixels"] == PIXEL)[0])
        donors = original["ids"][query].copy()
        weights = original["weights"][query].copy()
    assert np.array_equal(donors, [70673, 5902, 5903])
    source_xy, index_x, index_y = source_domain(vertices, old_tx, row)
    source_cells, source_corner_ids = touched_cells(vertices, source_xy)
    source_heights = vertices[source_corner_ids, 2].astype(float)
    zmin = min(float(old_tx[2]), float(np.floor(source_heights.min())+3.))
    zmax = max(float(old_tx[2]), float(source_heights.max()+3.))
    source = np.column_stack((source_xy, np.full(4, zmin)))
    rx_cells = np.array(sorted({divmod(int(face) % 65025, 255) for face in donors}),
                        dtype=np.int32)
    receiver_regions = []
    for r, c in rx_cells:
        corner_ids = np.array([r*256+c, r*256+c+1, (r+1)*256+c+1, (r+1)*256+c])
        receiver_regions.append(receiver_vertices[corner_ids].astype(float))
    receiver_regions = np.stack(receiver_regions)
    source_ground_clearance_lower = zmin-float(source_heights.max())
    assert source_ground_clearance_lower > 0
    saved = {
        "ground_vertices": vertices,
        "receiver_vertices": receiver_vertices,
        "main_faces": main,
        "anti_faces": anti,
        "source_rectangle_vertices": source,
        "source_xy": source_xy,
        "source_zmin": np.array(zmin),
        "source_zmax": np.array(zmax),
        "rx_cells": rx_cells,
        "rx_region_corners": receiver_regions,
        "donors": donors,
        "donor_weights": weights,
        "source_cells": source_cells,
        "source_corner_ids": source_corner_ids,
    }
    results = []
    for region_index, receiver in enumerate(receiver_regions):
        for mode, faces in (("main", main), ("anti", anti)):
            name = f"rx{region_index}_{mode}"
            record = certificate(vertices, faces, source, receiver, name, saved)
            record["rx_cell_row_col"] = rx_cells[region_index].tolist()
            record["ground_diagonal"] = mode
            results.append(record)
            print(json.dumps({key: record[key] for key in
                              ("name", "rx_cell_row_col", "minimum_gap_m", "positive_with_tolerance")}),
                  flush=True)
    all_positive = all(r["positive_with_tolerance"] for r in results)
    nested = []
    if not all_positive:
        old_source = old_tx[None, :]
        donor_triangles = receiver_vertices[main[donors]].astype(float)
        for stage, source_vertices, modes in (
            ("fixed_legacy_TX_original_donor_main", old_source, ("main",)),
            ("fixed_legacy_TX_original_donor_both", old_source, ("main", "anti")),
            ("TX_rectangle_original_donor_both", source, ("main", "anti")),
        ):
            records = []
            for donor_index, receiver in enumerate(donor_triangles):
                for mode in modes:
                    name = f"{stage}_donor{donor_index}_{mode}"
                    faces = main if mode == "main" else anti
                    records.append(certificate(vertices, faces, source_vertices, receiver,
                                               name, saved))
            nested.append({"stage": stage, "records": records,
                           "all_positive": all(r["positive_with_tolerance"] for r in records)})
    source_box_corners = np.concatenate((source, np.column_stack((source_xy, np.full(4, zmax)))))
    receiver_corners = receiver_regions.reshape(-1, 3)
    distances = np.linalg.norm(source_box_corners[:, None]-receiver_corners[None, :], axis=2)
    max_pair = np.unravel_index(np.argmax(distances), distances.shape)
    dmax = float(distances[max_pair])
    # Label-independent geometry is committed before reading labels or masks.
    np.savez_compressed(OUT/"certificate_geometry.npz", **saved)
    frequencies = np.array([415e6, 5.775e9])
    ideal_pl_upper = 20*np.log10(4*np.pi*frequencies*dmax/299792458.)
    labels = np.array([
        np.load(ROOT.parent/f"data/train_pl{band}.npy", mmap_mode="r")[MID].ravel()[PIXEL]
        for band in ("415", "58")], dtype=float)
    valid = [
        bool(np.unpackbits(np.load(ROOT.parent/f"data/train_mask{band}.npy",
                                   mmap_mode="r")[MID])[PIXEL])
        for band in ("415", "58")]
    assert all(valid) and np.array_equal(labels, [88.125, 110.6875])
    result = {
        "scope": "TRAIN 00414_00, map1819, target pixel(23,38), three frozen donor cells",
        "gpu_used": False,
        "numerical_method": "Double-precision ConvexHull lower facets; clip each candidate ground triangle against projected facet triangle; evaluate affine vertical gap at every intersection-polygon vertex.",
        "assumptions": [
            "Terrain vertices equal the published Float32 candidate reconstruction.",
            "Each ground grid cell may independently use either diagonal, with the same fixed Float32 vertices. Every triangle in both complete diagonal meshes was checked.",
            "Each RX point is anywhere on either triangulation of one of the three full donor cells shifted +1m along z.",
            "TX XY belongs to the stated rectangle, a hypothesis covering legacy coordinates and nearest/floor pixel-cell encodings under this candidate grid.",
            "For LOS only, TX z may be any value at least zmin. For the inverse-square power bound TX z is restricted to [zmin,zmax].",
            "The geometric LOS certificate does not guarantee numerical receiver-hit coverage or recover original scene coordinates.",
        ],
        "TX_domain": {
            "published_pixel_row_col": [69, 227],
            "pixel_encoding_index_col_interval": index_x,
            "pixel_encoding_index_row_interval": index_y,
            "legacy_world_TX": old_tx.tolist(),
            "xy_rectangle_corners_m": source_xy.tolist(),
            "zmin_m": zmin, "finite_power_zmax_m": zmax,
            "min_touched_terrain_corner_z_m": float(source_heights.min()),
            "max_touched_terrain_corner_z_m": float(source_heights.max()),
            "source_rectangle_ground_clearance_lower_bound_m": source_ground_clearance_lower,
            "touched_source_cells_row_col": source_cells.tolist(),
        },
        "RX_cells_row_col": rx_cells.tolist(),
        "source_donors": donors.tolist(), "source_weights": weights.tolist(),
        "all_expanded_domain_certificates_positive": all_positive,
        "minimum_certified_clearance_m": min(r["minimum_gap_m"] for r in results),
        "certificates": results, "nested_if_needed": nested,
        "clip_halfplane_tolerance_m2": CLIP_TOL,
        "positive_gap_tolerance_m": GAP_TOL,
        "minimum_witness_is_actual_ray_obstruction": False,
        "witness_limit": "A nonpositive hull lower-bound witness can be conservative and does not alone prove a realizable blocked TX/RX segment.",
        "conditional_ideal_area_direct_power": {
            "frequency_hz": frequencies.tolist(),
            "max_endpoint_distance_m": dmax,
            "max_distance_TX_world": source_box_corners[max_pair[0]].tolist(),
            "max_distance_RX_world": receiver_corners[max_pair[1]].tolist(),
            "normalized_inverse_square_gain_lower_bound": 1/dmax**2 if all_positive else None,
            "pathloss_upper_bound_dB": ideal_pl_upper.tolist() if all_positive else None,
            "published_label_pathloss_dB": labels.tolist(),
            "minimum_direct_power_excess_above_label_dB": (labels-ideal_pl_upper).tolist() if all_positive else None,
            "limit": "Ideal geometric direct-area power with isotropic free-space normalization. Does not bound native numerical receiver stepping or a different release support.",
        },
        "sources": {
            "c141_replay_sha256": sha256(c141_path),
            "c142_geometry_source_sha256": sha256(ROOT/"src/c142_backend.py"),
            "published_height_sample_sha256": hashlib.sha256(np.ascontiguousarray(height).tobytes()).hexdigest(),
            "candidate_ground_ply_sha256": sha256(ROOT/"audit/c142_backend/inputs/1819/ground.ply"),
            "candidate_receiver_ply_sha256": sha256(ROOT/"audit/c142_backend/inputs/1819/rx.ply"),
            "script_sha256": sha256(Path(__file__).resolve()),
            "geometry_npz_sha256": sha256(OUT/"certificate_geometry.npz"),
        },
    }
    write_json(OUT/"result.json", result)
    print(json.dumps({
        "all_positive": all_positive,
        "minimum_clearance_m": result["minimum_certified_clearance_m"],
        "TX_domain": result["TX_domain"],
        "ideal_area_direct": result["conditional_ideal_area_direct_power"],
    }), flush=True)


if __name__ == "__main__":
    main()
