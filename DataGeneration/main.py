
"""
main.py — Orchestrates the OpenLiver simulation using YAML config.

Modules it uses:
  - sofa_scene.py  : create_scene (SOFA graph, controller, regions)
  - raycast.py     : build_camera_rig, intrinsics_from_cfg, perform_raycast
  - io_utils.py    : folder setup, metadata, frame saving, checkpoints & completeness

Run:
  python main.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import re
import json
import random
from typing import Dict, Tuple

import numpy as np
import yaml
import open3d as o3d
import Sofa
import Sofa.Core
import Sofa.Simulation

from sofa_scene import create_scene
from raycast import build_camera_rig, intrinsics_from_cfg, perform_raycast
from io_utils import(
    setup_liver_simulation_folder,
    write_metadata,
    save_frame_npz,
    frame_is_complete,
    save_checkpoint,
    load_latest_checkpoint,
    sim_is_complete,)



# ----------------------------- utilities ---------------------------------

#def _seed_for(liver_id: int, sim_id: int, base_seed: int, enable_per_liver_sim: bool) -> int:
#    if enable_per_liver_sim:
#        return (int(base_seed) * 73856093 + liver_id * 19349663 + sim_id * 83492791) & 0x7FFFFFFF
#    return int(base_seed)


def _sample_youngs(cfg: Dict, rng: random.Random) -> int:
    lo, hi = cfg["physics"]["young_modulus_range"]
    step = max(1, int(cfg["physics"].get("young_modulus_step", 1)))
    # inclusive range like your original `randrange(2000, 5001, 200)`
    choices = list(range(int(lo), int(hi) + 1, step))
    return rng.choice(choices)


def _digits_in_name(path: str) -> int:
    """Extract first integer from file name, fallback to 0."""
    base = os.path.basename(path)
    m = re.search(r"(\d+)", base)
    return int(m.group(1)) if m else 0


# ----------------------------- main runner --------------------------------

def run(cfg: Dict):
    # Paths
    livers_folder = cfg["paths"]["livers"]
    surfaces_folder = cfg["paths"]["surfaces"]
    output_base = cfg["paths"]["out"]

    frames = int(cfg["run"]["frames"])
    dt = float(cfg["run"]["dt"])
    sims_per_liver = int(cfg["run"]["simulations_per_liver"])
    resume = bool(cfg["run"].get("resume", True))
    ckpt_every = int(cfg["run"].get("checkpoint_every", 10))

    # Physics core (constant parts)
    poisson = float(cfg["physics"]["poisson_ratio"])
    total_mass = float(cfg["physics"]["total_mass"])
    rayleigh_stiff = float(cfg["physics"]["rayleigh_stiffness"])
    rayleigh_mass = float(cfg["physics"]["rayleigh_mass"])
    cycle_len = int(cfg["forces"]["cycle_length"])

    # Cameras
    radii = [float(r) for r in cfg["cameras"]["radii"]]
    angles = [float(a) for a in cfg["cameras"]["angles_deg"]]
    y_offset = float(cfg["cameras"]["y_offset"])
    sensors = cfg["cameras"]["sensors"]  # list of dicts with name/K/resolution

    # Noise
    noise_mode = str(cfg["noise"].get("mode", "ray_aligned"))
    noise_max_sigma_m = float(cfg["noise"].get("max_sigma_m", 0.005))

    # Saving options
    disp_thresh = float(cfg["saving"]["displacement_threshold_m"])
    tre_samples = int(cfg["saving"]["tre_samples"])

    # Repro
    base_seed = int(cfg["seeds"].get("base", 1234))
    seed_per_liver_sim = bool(cfg["seeds"].get("per_liver_sim", True))

    # Discover livers
    liver_mesh_files = sorted(
        [os.path.join(livers_folder, f) for f in os.listdir(livers_folder) if f.lower().endswith(".msh")]
    )
    if not liver_mesh_files:
        raise FileNotFoundError(f"No .msh files in {livers_folder}")

    for liver_mesh_path in liver_mesh_files:
        liver_id = _digits_in_name(liver_mesh_path)
        surface_path = os.path.join(surfaces_folder, f"liver_{liver_id}.nii_mesh.stl")
        if not os.path.exists(surface_path):
            # fallback: try liver_{id}.stl
            alt = os.path.join(surfaces_folder, f"liver_{liver_id}.stl")
            if os.path.exists(alt):
                surface_path = alt
            else:
                print(f"[warn] Missing surface mesh for liver {liver_id}: {surface_path}")
                continue

        print(f"\n=== Liver {liver_id:03d} ===")

        liver_root_on_disk = os.path.join(output_base, f"Liver_{liver_id:03d}")
        os.makedirs(liver_root_on_disk, exist_ok=True)

        for sim_id in range(1, sims_per_liver + 1):
            sim_root = os.path.join(liver_root_on_disk, f"Simulation_{sim_id:02d}")
            if os.path.exists(sim_root) and sim_is_complete(
                sim_root, sensors, frames, angles, radii
            ):
                print(f"[skip] Liver {liver_id:03d} Sim {sim_id:02d} already complete.")
                continue

            # Seeding (reproducible per liver+sim)
            sim_seed = liver_id * 10000 + sim_id
            random.seed(sim_seed)
            np.random.seed(sim_seed)



            # Young's modulus sampled per simulation
            ym = _sample_youngs(cfg, random)

            # Build SOFA scene
            root = Sofa.Core.Node("root")
            # Inject required runtime fields into cfg for this sim:
            cfg_local = json.loads(json.dumps(cfg))  # shallow deep-copy dict-of-lists
            cfg_local["run"]["dt"] = dt
            cfg_local["physics"]["poisson_ratio"] = poisson
            cfg_local["physics"]["total_mass"] = total_mass
            cfg_local["physics"]["rayleigh_stiffness"] = rayleigh_stiff
            cfg_local["physics"]["rayleigh_mass"] = rayleigh_mass
            cfg_local["physics"]["young_modulus"] = ym
            cfg_local["forces"]["cycle_length"] = cycle_len

            liverNode, visuNode, topo, controller = create_scene(
                rootNode=root,
                liver_mesh_path=liver_mesh_path,
                liver_surface_path=surface_path,
                cfg=cfg_local,
            )

            # Scene init
            Sofa.Simulation.init(root)

            # Base states
            visu_model = visuNode.getObject("VisualModel")
            vis_undeformed_o = np.asarray(visu_model.position.value)
            vis_undeformed= np.asarray(vis_undeformed_o, dtype=np.float64).copy(order="C")

            surface_faces_o = np.asarray(visu_model.triangles.value)
            surface_faces = np.asarray(surface_faces_o, dtype=np.int32).copy(order="C")
            liver_dofs = liverNode.getObject("dofs")
            vol_undeformed_o = np.asarray(liver_dofs.position.value)
            vol_undeformed = np.asarray(vol_undeformed_o, dtype=np.float64).copy(order="C")
            tetrahedra_indices = np.asarray(liverNode.getObject("topo").tetrahedra.value)

            # Centroid for rig
            centroid = np.mean(vol_undeformed, axis=0)

            # Camera rig & folders
            cam_rig = build_camera_rig(centroid, radii=radii, angles_deg=angles, y_offset=y_offset)
            liver_folder, sim_folder, cam_map = setup_liver_simulation_folder(
                out_base=output_base,
                liver_id=liver_id,
                sim_id=sim_id,
                camera_rig=cam_rig,
                sensors=sensors,
            )

            # Force metadata (per-region direction & magnitude)
            force_profiles_dict = {
                region: {
                    "direction": controller.force_profiles[region].tolist(),
                    "max_magnitude": float(controller.max_magnitudes[region]),
                    "force_per_node": float(controller.max_magnitudes[region]) / max(1, len(controller.force_regions[region])),
                }
                for region in controller.force_profiles
            }

            # Write metadata.json
            write_metadata(sim_folder, {
                "liver_id": liver_id,
                "simulation_id": sim_id,
                "young_modulus": ym,
                "dt": dt,
                "frames": frames,
                "fixed_parts": controller.fixed_parts,
                "force_parts": controller.force_parts,
                "fixed_nodes": controller.fixed_nodes,
                "force_nodes": controller.force_nodes,
                "force_profiles": force_profiles_dict,
                "cameras": {
                    "radii": radii,
                    "angles_deg": angles,
                    "y_offset": y_offset,
                    "sensors": sensors,
                },
                "noise": {
                    "mode": noise_mode,
                    "max_sigma_m": noise_max_sigma_m,
                },
            })

            # --- TRE probe selection (K random tetra centroids) ---
            t_count = tetrahedra_indices.shape[0]
            K_tre = min(tre_samples, t_count) if t_count > 0 else 0
            rng_np = np.random.default_rng(seed=42 + liver_id * 100 + sim_id)
            tre_tetra_ids = rng_np.choice(t_count, size=K_tre, replace=False) if K_tre > 0 else None

            def _tetra_centroids(verts_3d, tets):
                return np.mean(verts_3d[tets], axis=1)

            tre_points_undeformed = (
                _tetra_centroids(vol_undeformed, tetrahedra_indices[tre_tetra_ids]) if K_tre > 0 else None
            )

            # --- Precompute undeformed partials & undeformed correspondences ---
            partial_undeformed_per_cam_sensor: Dict[Tuple[float, float, str], np.ndarray] = {}
            precomputed_corr_u_per_cam_sensor: Dict[Tuple[float, float, str], np.ndarray] = {}


            #####################
            #pts = np.asarray(vis_undeformed, dtype=np.float64).copy(order="C")  # writeable, C-contiguous
            #if pts.ndim != 2 or pts.shape[1] != 3:
            #    raise ValueError(f"Expected (N, 3) points, got {pts.shape}")

            #pcd = o3d.geometry.PointCloud()
            #pcd.points = o3d.utility.Vector3dVector(pts)
            #kd_full_u = o3d.geometry.KDTreeFlann(pcd)

            kd_full_u = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vis_undeformed)))

            for cam in cam_rig:
                cam_pos = cam["pos"]
                angle = float(cam["angle_deg"])
                radius = float(cam["radius"])

                for sensor in sensors:
                    fx, fy, cx, cy, res = intrinsics_from_cfg(sensor)
                    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

                    pu, _ = perform_raycast(
                        visuNode=visuNode,
                        camera_pos=cam_pos,
                        centroid=centroid,
                        K=K,
                        resolution=res,
                    )
                    partial_undeformed_per_cam_sensor[(angle, radius, sensor["name"])] = pu

                    # nearest-neighbor indices on undeformed full surface
                    corr = []
                    for p in pu:
                        _, ind, _ = kd_full_u.search_knn_vector_3d(p, 1)
                        corr.append(ind[0])
                    precomputed_corr_u_per_cam_sensor[(angle, radius, sensor["name"])] = np.asarray(corr, dtype=np.int32)

            # --- Resume from checkpoint if any ---
            Sofa.Simulation.init(root)  # ensure graph is ready before any restore
            if resume and os.path.exists(sim_folder):
                loaded, last_frame_ck, prev_pc_ck = load_latest_checkpoint(sim_folder, root, liverNode, controller)
            else:
                loaded, last_frame_ck, prev_pc_ck = (False, -1, None)

            if loaded:
                start_frame = last_frame_ck + 1
                prev_pc = prev_pc_ck if prev_pc_ck is not None else np.asarray(visu_model.position.value).copy()
                print(f"[resume] Resuming at frame {start_frame}.")
            else:
                start_frame = 0
                prev_pc = vis_undeformed.copy()
                print("[start] Starting from frame 0.")

            # ----------------------- main frame loop -----------------------
            for frame in range(start_frame, frames):
                # advance physics by dt
                Sofa.Simulation.animate(root, float(dt))

                # fetch current deformed states
                vis_deformed = np.asarray(visu_model.position.value)
                vol_deformed = np.asarray(liver_dofs.position.value)

                # iterate cameras & sensors
                for cam in cam_rig:
                    cam_pos = cam["pos"]
                    angle = float(cam["angle_deg"])
                    radius = float(cam["radius"])
                    cam_root = os.path.join(sim_folder, f"Camera_angle{int(angle)}".replace('-', 'N') + f"_radius{radius:.2f}".replace('.', 'p'))

                    for sensor in sensors:
                        sname = sensor["name"]
                        s_dir = os.path.join(cam_root, sname)
                        frame_path = os.path.join(s_dir, f"frame_{frame:04d}.npz")
                        if os.path.exists(frame_path):
                            continue  # already saved (e.g., after crash mid-frame)

                        fx, fy, cx, cy, res = intrinsics_from_cfg(sensor)
                        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

                        # raycast deformed partial + directions
                        pd, hit_dirs = perform_raycast(
                            visuNode=visuNode,
                            camera_pos=cam_pos,
                            centroid=centroid,
                            K=K,
                            resolution=res,
                        )

                        pu = partial_undeformed_per_cam_sensor[(angle, radius, sname)]
                        pre_corr_u = precomputed_corr_u_per_cam_sensor[(angle, radius, sname)]

                        # save NPZ
                        save_frame_npz(
                            camera_folder=s_dir,
                            frame=frame,
                            full_pc_undeformed=vis_undeformed,
                            full_pc_deformed=vis_deformed.copy(),
                            full_pc_previous=prev_pc.copy(),
                            surface_faces_original=surface_faces.copy(),
                            partial_pc_undeformed=pu.copy(),
                            partial_pc_deformed=pd.copy(),
                            ray_dirs_world=hit_dirs,
                            tetrahedra_indices=tetrahedra_indices.copy(),
                            tetrahedra_vertices_deformed=vol_deformed.copy(),
                            tetrahedra_vertices_undeformed=vol_undeformed.copy(),
                            tre_tetra_ids=tre_tetra_ids,
                            tre_points_undeformed=tre_points_undeformed,
                            noise_mode=noise_mode,
                            noise_max_sigma_m=noise_max_sigma_m,
                            displacement_threshold_m=disp_thresh,
                            precomputed_corr_u=pre_corr_u,
                            force_node_indices=[idx for region_nodes in controller.force_nodes.values() for idx in region_nodes],
                        )

                # update prev surface for step_flow
                prev_pc = vis_deformed.copy()

                # checkpoint (if whole frame finished)
                if frame_is_complete(sim_folder, frame, sensors, angles, radii):
                    if ((frame % ckpt_every) == 0) or (frame == frames - 1):
                        save_checkpoint(sim_folder, frame, liverNode, controller, prev_pc)

            print(f"[done] Liver {liver_id:03d} Simulation {sim_id:02d} finished.")

    print("\nAll requested livers/simulations processed.")


# ----------------------------- entrypoint ---------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="OpenLiver Simulation Orchestrator")
    ap.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to YAML config.")
    return ap.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # basic sanity defaults
    cfg.setdefault("run", {}).setdefault("resume", True)
    cfg.setdefault("run", {}).setdefault("checkpoint_every", 10)
    return cfg


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    run(config)
