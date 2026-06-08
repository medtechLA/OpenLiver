
"""
io_utils.py — paths, saving, metadata, completeness checks, and checkpoints
"""

from __future__ import annotations

import os
import json
import glob
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import open3d as o3d


# -------------------------------------------------------------------------
# --- Logging -----------------------------------------------
# -------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


# -------------------------------------------------------------------------
# --- Path helpers ---------------------------------------------------------
# -------------------------------------------------------------------------

def camera_folder_name(angle_deg: float, radius: float) -> str:
    angle_str = f"{int(angle_deg)}".replace('-', 'N')
    radius_str = f"{radius:.2f}".replace('.', 'p')
    return f"Camera_angle{angle_str}_radius{radius_str}"


def expected_camera_folders(camera_angles_deg: List[float], camera_radii: List[float]) -> List[str]:
    return [camera_folder_name(a, r) for a in camera_angles_deg for r in camera_radii]


def setup_liver_simulation_folder(
    out_base: str,
    liver_id: int,
    sim_id: int,
    camera_rig: List[Dict],
    sensors: List[Dict],
) -> Tuple[str, str, Dict[Tuple[float, float], str]]:
    """
    Create folders:
      <out_base>/Liver_XXX/Simulation_YY/Camera_angleA_radiusR/{SENSOR_NAME}
    Returns:
      (liver_root, sim_root, {(angle, radius): cam_root})
    """
    liver_root = os.path.join(out_base, f"Liver_{liver_id:03d}")
    sim_root = os.path.join(liver_root, f"Simulation_{sim_id:02d}")
    os.makedirs(sim_root, exist_ok=True)

    cam_map: Dict[Tuple[float, float], str] = {}
    for c in camera_rig:
        angle, radius = float(c["angle_deg"]), float(c["radius"])
        cam_root = os.path.join(sim_root, camera_folder_name(angle, radius))
        os.makedirs(cam_root, exist_ok=True)
        for s in sensors:
            s_name = s["name"]
            os.makedirs(os.path.join(cam_root, s_name), exist_ok=True)
        cam_map[(angle, radius)] = cam_root

    return liver_root, sim_root, cam_map


# -------------------------------------------------------------------------
# --- Completeness checks --------------------------------------------------
# -------------------------------------------------------------------------

def frame_is_complete(
    sim_folder: str,
    frame_idx: int,
    sensors: List[Dict],
    camera_angles_deg: List[float],
    camera_radii: List[float],
) -> bool:
    """Return True iff every expected camera/sensor has frame_{frame_idx:04d}.npz."""
    fname = f"frame_{frame_idx:04d}.npz"
    for cam_name in expected_camera_folders(camera_angles_deg, camera_radii):
        cam_root = os.path.join(sim_folder, cam_name)
        if not os.path.isdir(cam_root):
            return False
        for s in sensors:
            s_name = s["name"]
            s_dir = os.path.join(cam_root, s_name)
            if not os.path.isdir(s_dir):
                return False
            if not os.path.exists(os.path.join(s_dir, fname)):
                return False
    return True


def sim_is_complete(
    sim_folder: str,
    sensors: List[Dict],
    simulation_frames: int,
    camera_angles_deg: List[float],
    camera_radii: List[float],
) -> bool:
    """Return True if all expected camera/sensor folders exist and have all frames."""
    for cam_name in expected_camera_folders(camera_angles_deg, camera_radii):
        cam_root = os.path.join(sim_folder, cam_name)
        if not os.path.isdir(cam_root):
            return False
        for s in sensors:
            s_name = s["name"]
            s_dir = os.path.join(cam_root, s_name)
            if not os.path.isdir(s_dir):
                return False
            existing = len(glob.glob(os.path.join(s_dir, "frame_*.npz")))
            if existing < simulation_frames:
                return False
    return True


# -------------------------------------------------------------------------
# --- Metadata -------------------------------------------------------------
# -------------------------------------------------------------------------

def write_metadata(sim_folder: str, metadata: Dict) -> None:
    path = os.path.join(sim_folder, "metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"[meta] Wrote {path}")


# -------------------------------------------------------------------------
# --- Checkpoints (resume support) ----------------------------------------
# -------------------------------------------------------------------------

def save_checkpoint(sim_folder: str, frame_idx: int, liverNode, controller, prev_pc: Optional[np.ndarray]) -> None:
    """
    Save a resume snapshot:
      - positions/velocities,
      - controller frame counter,
      - prev_pc (for step_flow),
      - Python & NumPy RNG states.
    """
    ck_dir = os.path.join(sim_folder, "_checkpoints")
    os.makedirs(ck_dir, exist_ok=True)

    dofs = liverNode.getObject("dofs")
    pos = np.array(dofs.position.value)
    try:
        vel = np.array(dofs.velocity.value)
    except Exception:
        vel = np.zeros_like(pos)

    py_rand_state = random.getstate()
    np_rand_state = np.random.get_state()

    np.savez_compressed(
        os.path.join(ck_dir, f"state_frame_{frame_idx:04d}.npz"),
        frame_idx=frame_idx,
        dofs_pos=pos,
        dofs_vel=vel,
        controller_frame_count=np.int64(controller.frame_count),
        prev_pc=prev_pc if prev_pc is not None else None,
        py_rand_state=np.array(py_rand_state, dtype=object),
        np_rand_state=np.array(np_rand_state, dtype=object),
    )
    log(f"Checkpoint saved at frame {frame_idx}.")


def load_latest_checkpoint(sim_folder: str, root, liverNode, controller) -> Tuple[bool, int, Optional[np.ndarray]]:
    """
    Load the most recent checkpoint if present.
    Returns (loaded, last_frame, prev_pc).
    """
    ck_dir = os.path.join(sim_folder, "_checkpoints")
    if not os.path.isdir(ck_dir):
        return (False, -1, None)

    ckpts = sorted(glob.glob(os.path.join(ck_dir, "state_frame_*.npz")))
    if not ckpts:
        return (False, -1, None)

    last_ck = ckpts[-1]
    data = np.load(last_ck, allow_pickle=True)
    frame_idx = int(data["frame_idx"])
    pos = data["dofs_pos"]
    vel = data["dofs_vel"]
    ctrl_fc = int(data["controller_frame_count"])
    prev_pc = data["prev_pc"] if "prev_pc" in data.files else None

    # restore RNG
    try:
        random.setstate(tuple(data["py_rand_state"]))
        np.random.set_state(tuple(data["np_rand_state"]))
    except Exception:
        pass

    # restore SOFA state
    dofs = liverNode.getObject("dofs")
    dofs.position.value = pos.tolist()
    try:
        dofs.velocity.value = vel.tolist()
    except Exception:
        pass

    controller.frame_count = ctrl_fc

    # propagate mapping/visuals without advancing time
    try:
        import Sofa
        Sofa.Simulation.animate(root, 0.0)
    except Exception:
        try:
            import Sofa
            Sofa.Simulation.updateVisual(root)
        except Exception:
            pass

    log(f"[resume] Loaded checkpoint {os.path.basename(last_ck)}")
    return (True, frame_idx, prev_pc)


# -------------------------------------------------------------------------
# --- Saving one frame (NPZ schema) ---------------------------------------
# -------------------------------------------------------------------------

def _nearest_indices(pts_src: np.ndarray, pts_tgt: np.ndarray) -> np.ndarray:
    """
    For each point in pts_src, find index of NN in pts_tgt.
    Returns (N_src,) int array of target indices.
    """
    if len(pts_src) == 0 or len(pts_tgt) == 0:
        return np.zeros((0,), dtype=np.int32)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_tgt))
    kdt = o3d.geometry.KDTreeFlann(pcd)
    idxs = np.empty((len(pts_src),), dtype=np.int32)
    for i, p in enumerate(pts_src):
        _, ind, _ = kdt.search_knn_vector_3d(p, 1)
        idxs[i] = ind[0]
    return idxs


def save_frame_npz(
    camera_folder: str,
    frame: int,
    *,
    # full (surface)
    full_pc_undeformed: np.ndarray,
    full_pc_deformed: np.ndarray,
    full_pc_previous: Optional[np.ndarray],
    surface_faces_original: np.ndarray,

    # partials
    partial_pc_undeformed: np.ndarray,
    partial_pc_deformed: np.ndarray,

    # ray info (for noise)
    ray_dirs_world: Optional[np.ndarray] = None,

    # tetra / TRE
    tetrahedra_indices: Optional[np.ndarray] = None,
    tetrahedra_vertices_deformed: Optional[np.ndarray] = None,
    tetrahedra_vertices_undeformed: Optional[np.ndarray] = None,
    tre_tetra_ids: Optional[np.ndarray] = None,
    tre_points_undeformed: Optional[np.ndarray] = None,

    # options
    noise_mode: str = "ray_aligned",  # "ray_aligned" | "z_only" | "off"
    noise_max_sigma_m: float = 0.005,
    displacement_threshold_m: float = 0.05,

    # precomputed
    precomputed_corr_u: Optional[np.ndarray] = None,

    # extras
    force_node_indices: Optional[np.ndarray] = None,
) -> str:
    """
    Write a single frame NPZ with:
      - full/partial pcs (+ noisy partial),
      - flow_full, step_flow,
      - correspondences (pairs) & visibility,
      - volumetric tets + TRE points.
    """
    fu = np.asarray(full_pc_undeformed, dtype=np.float32)
    fd = np.asarray(full_pc_deformed, dtype=np.float32)
    fp = np.asarray(full_pc_previous, dtype=np.float32) if full_pc_previous is not None else None

    pu = np.asarray(partial_pc_undeformed, dtype=np.float32)
    pd = np.asarray(partial_pc_deformed, dtype=np.float32)

    # --- Outlier handling on full (deformed vs undeformed) ---
    disp = np.linalg.norm(fd - fu, axis=1)
    outliers = disp > displacement_threshold_m
    kept_mask = ~outliers
    do_filter = 0 < outliers.sum() <= 5  # small #outliers -> filter, else keep as-is

    if do_filter:
        fu_f = fu[kept_mask]
        fd_f = fd[kept_mask]
        fp_f = fp[kept_mask] if fp is not None else None
        idx_map = -np.ones(len(kept_mask), dtype=np.int32)
        idx_map[kept_mask] = np.arange(kept_mask.sum(), dtype=np.int32)
        log(f"⚠ Frame {frame}: Removing {outliers.sum()} outliers (>{displacement_threshold_m:.3f} m)")
    else:
        fu_f, fd_f, fp_f = fu, fd, fp
        idx_map = np.arange(len(fu), dtype=np.int32)

    # --- Flow (filtered arrays) ---
    flow_full = fd_f - fu_f
    step_flow = (fd_f - fp_f) if fp_f is not None else np.zeros_like(fd_f)

    # --- Correspondences (deformed path): partial_d -> full_d (filtered) ---
    corr_d = _nearest_indices(pd, fd_f)
    corr_pairs_deformed = np.column_stack([
        np.arange(len(pd), dtype=np.int32), corr_d
    ]) if len(pd) else np.zeros((0, 2), dtype=np.int32)

    # --- Correspondences (undeformed path): partial_u -> full_u (filtered) ---
    if precomputed_corr_u is not None and len(precomputed_corr_u) == len(pu):
        corr_u = np.asarray(precomputed_corr_u, dtype=np.int32)
        if do_filter:
            valid = (corr_u >= 0) & (corr_u < len(idx_map)) & (idx_map[corr_u] >= 0)
            pu = pu[valid]
            corr_u = idx_map[corr_u[valid]]
    else:
        corr_u = _nearest_indices(pu, fu_f)
    corr_pairs_undeformed = np.column_stack([
        np.arange(len(pu), dtype=np.int32), corr_u
    ]) if len(pu) else np.zeros((0, 2), dtype=np.int32)

    # --- Visibility (unique coverage on full surface) ---
    unique_vis_d = int(np.unique(corr_pairs_deformed[:, 1]).size) if corr_pairs_deformed.size else 0
    unique_vis_u = int(np.unique(corr_pairs_undeformed[:, 1]).size) if corr_pairs_undeformed.size else 0
    visibility_ratio_surface_deformed = float(unique_vis_d) / max(1, fd_f.shape[0])
    visibility_ratio_surface_undeformed = float(unique_vis_u) / max(1, fu_f.shape[0])

    # --- Noise on partial deformed (ray-aligned or z-only) ---
    if noise_mode.lower() != "off" and len(pd) > 0 and (noise_max_sigma_m > 0):
        sigmas = np.random.uniform(0.0, float(noise_max_sigma_m), size=len(pd)).astype(pd.dtype)
        if noise_mode == "ray_aligned" and ray_dirs_world is not None and len(ray_dirs_world) == len(pd):
            rays = np.asarray(ray_dirs_world, dtype=pd.dtype)
            rays /= (np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12)
            n = np.random.normal(0.0, 1.0, size=len(pd)).astype(pd.dtype)
            pd_noisy = pd + (sigmas[:, None] * n[:, None] * rays)
            average_ray_noise_magnitude = float(np.mean(np.abs(sigmas)))
            ray_noise_std = float(np.std(sigmas))
            average_z_noise_magnitude = 0.0
            z_noise_std = 0.0
        else:
            n = np.random.normal(0.0, sigmas).astype(pd.dtype)
            pd_noisy = pd.copy()
            pd_noisy[:, 2] += n
            average_ray_noise_magnitude = 0.0
            ray_noise_std = 0.0
            average_z_noise_magnitude = float(np.mean(np.abs(n)))
            z_noise_std = float(np.std(n))
    else:
        pd_noisy = pd
        average_ray_noise_magnitude = 0.0
        ray_noise_std = 0.0
        average_z_noise_magnitude = 0.0
        z_noise_std = 0.0

    # --- TRE probes (interior centroids, per-frame deformed) ---
    tre_points_deformed = None
    if (tre_tetra_ids is not None) and (tetrahedra_vertices_deformed is not None) and (tetrahedra_indices is not None):
        tsel = tetrahedra_indices[tre_tetra_ids]  # (K,4)
        tre_points_deformed = np.mean(tetrahedra_vertices_deformed[tsel], axis=1)  # (K,3)

    # --- Mesh dicts (keep faces unchanged) ---
    surface_mesh_undeformed = {'vertices': fu, 'triangles': surface_faces_original}
    surface_mesh_deformed = {'vertices': fd, 'triangles': surface_faces_original}

    # --- Save ---
    save_path = os.path.join(camera_folder, f"frame_{frame:04d}.npz")
    np.savez_compressed(
        save_path,
        # full (unfiltered + filtered)
        full_pc_undeformed=fu,
        full_pc_deformed=fd,
        full_pc_previous=fp,
        full_pc_undeformed_filtered=fu_f,
        full_pc_deformed_filtered=fd_f,
        full_pc_previous_filtered=fp_f,
        kept_mask=kept_mask,

        # partials
        partial_pc_undeformed=pu,
        partial_pc_deformed=pd,
        partial_pc_deformed_noisy=pd_noisy,

        # flow
        flow_full=flow_full,
        step_flow=step_flow,

        # mesh (topology consistent)
        surface_mesh_undeformed=surface_mesh_undeformed,
        surface_mesh_deformed=surface_mesh_deformed,

        # correspondences
        corr_pairs_deformed=corr_pairs_deformed,
        corr_pairs_undeformed=corr_pairs_undeformed,
        partial_to_full_correspondences=corr_d,
        partial_to_full_correspondences_undeformed=corr_u,

        # visibility
        visibility_ratio_surface_deformed=visibility_ratio_surface_deformed,
        visibility_ratio_surface_undeformed=visibility_ratio_surface_undeformed,

        # volume + TRE
        tetrahedra_indices=tetrahedra_indices,
        tetrahedra_vertices_deformed=tetrahedra_vertices_deformed,
        tetrahedra_vertices_undeformed=tetrahedra_vertices_undeformed,
        tre_tetra_ids=tre_tetra_ids,
        tre_points_undeformed=tre_points_undeformed,
        tre_points_deformed=tre_points_deformed,

        # extras
        force_node_indices=np.asarray(force_node_indices) if force_node_indices is not None else None,

        # noise stats (for reproducibility)
        average_z_noise=average_z_noise_magnitude,
        z_noise_std=z_noise_std,
        average_ray_noise=average_ray_noise_magnitude,
        ray_noise_std=ray_noise_std,
    )
    log(f" Saved frame {frame} → {save_path}")
    return save_path
