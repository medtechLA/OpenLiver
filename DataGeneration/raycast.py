
"""
raycast.py — camera rigs, intrinsics, and Open3D-based raycasting
"""

from __future__ import annotations

import os
import numpy as np
from typing import Dict, List, Tuple, Optional

import open3d as o3d
import open3d.core as o3c


# -------------------------------------------------------------------------
# --- Intrinsics utilities -------------------------------------------------
# -------------------------------------------------------------------------

def intrinsics_from_cfg(sensor_cfg: Dict) -> Tuple[float, float, float, float, Tuple[int, int]]:
    """
    Parse one sensor entry from YAML config and return (fx, fy, cx, cy, (W,H)).
    Expected shape:
      sensor_cfg = {
        "name": "HL",
        "resolution": [W, H],
        "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
      }
    """
    K = np.asarray(sensor_cfg["K"], dtype=float)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    res = tuple(int(x) for x in sensor_cfg["resolution"])  # (W, H)
    return fx, fy, cx, cy, res


# -------------------------------------------------------------------------
# --- Camera rig (positions around centroid) -------------------------------
# -------------------------------------------------------------------------

def build_camera_rig(
    centroid: np.ndarray,
    radii: List[float],
    angles_deg: List[float],
    y_offset: float = 0.25,
) -> List[Dict]:
    """
    Build a simple ring rig: for each radius and angle, place a camera at (radius, angle)
    around the centroid, then add an 'up' offset orthogonal to the view direction.

    Returns a list of dicts:
      [{ "pos": (3,), "angle_deg": a, "radius": r }, ...]
    """
    centroid = np.asarray(centroid, dtype=float)
    rig = []
    for r in radii:
        for a_deg in angles_deg:
            a = np.radians(a_deg)
            base = centroid + np.array([np.sin(a) * r, 0.0, np.cos(a) * r], dtype=float)

            view_dir = centroid - base
            n = np.linalg.norm(view_dir)
            view_dir = view_dir / n if n > 0 else np.array([0, 0, 1], dtype=float)

            # project y_offset onto plane orthogonal to 'view_dir'
            up_vec = np.array([0, y_offset, 0], dtype=float)
            up_proj = up_vec - np.dot(up_vec, view_dir) * view_dir
            cam_pos = base + up_proj

            rig.append({"pos": cam_pos, "angle_deg": a_deg, "radius": r})
    return rig


# -------------------------------------------------------------------------
# --- Pixel directions cache ----------------------------------------------
# -------------------------------------------------------------------------

# cache key: (W, H, fx, fy, cx, cy)
_PIXEL_DIR_CACHE: Dict[Tuple[int, int, float, float, float, float], np.ndarray] = {}


def get_pixel_directions(resolution: Tuple[int, int], fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Vectorized pinhole rays in camera frame (unit vectors), sampled at pixel centers.
    Returns array of shape (H*W, 3) as float32.
    """
    W, H = int(resolution[0]), int(resolution[1])
    key = (W, H, float(fx), float(fy), float(cx), float(cy))
    cached = _PIXEL_DIR_CACHE.get(key, None)
    if cached is not None:
        return cached

    xs = np.arange(W, dtype=np.float32) + 0.5
    ys = np.arange(H, dtype=np.float32) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)  # (H, W)

    dir_x = (grid_x - cx) / fx
    dir_y = (grid_y - cy) / fy
    dir_z = np.ones_like(dir_x, dtype=np.float32)

    dirs = np.stack([dir_x, dir_y, dir_z], axis=-1).reshape(-1, 3).astype(np.float32)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(norms, 1e-8).astype(np.float32)
    _PIXEL_DIR_CACHE[key] = dirs
    return dirs


# -------------------------------------------------------------------------
# --- Look-at rotation -----------------------------------------------------
# -------------------------------------------------------------------------

def robust_look_at_rotation(camera_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """
    Build a right-handed look-at rotation (columns: right, up, forward),
    stable for near-parallel up/forward.
    """
    cam = np.asarray(camera_pos, dtype=float)
    tgt = np.asarray(target_pos, dtype=float)

    fwd = tgt - cam
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 0 else np.array([0, 0, 1], dtype=float)

    up = np.array([0, 1, 0], dtype=float)
    if abs(np.dot(up, fwd)) > 0.99:
        up = np.array([0, 0, 1], dtype=float)

    right = np.cross(up, fwd)
    nr = np.linalg.norm(right)
    right = right / (nr if nr > 0 else 1.0)

    up_corr = np.cross(fwd, right)
    nu = np.linalg.norm(up_corr)
    up_corr = up_corr / (nu if nu > 0 else 1.0)

    # Columns (right, up, forward)
    R = np.vstack([right, up_corr, fwd]).T  # (3,3)
    return R.astype(np.float32)


# -------------------------------------------------------------------------
# --- Open3D raycasting helpers -------------------------------------------
# -------------------------------------------------------------------------

def _build_o3d_ray_scene_from_visu(visuNode) -> Tuple[o3d.t.geometry.RaycastingScene, o3d.geometry.TriangleMesh]:
    """
    Create an Open3D t::RaycastingScene from the SOFA visu node's surface.
    """
    visu_mesh = visuNode.getObject('VisualModel')
    vertices = np.asarray(visu_mesh.position.value, dtype=np.float32)
    faces_o = np.asarray(visu_mesh.triangles.value, dtype=np.int32)
    faces =  np.asarray(faces_o, dtype=np.int32).copy(order="C")

    legacy_mesh = o3d.geometry.TriangleMesh()
    legacy_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    legacy_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # build t-mesh for ray scene
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    return scene, legacy_mesh


def perform_raycast(
    visuNode,
    camera_pos: np.ndarray,
    centroid: np.ndarray,
    K: np.ndarray,
    resolution: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized ray casting using Open3D RaycastingScene against the Visu surface.

    Inputs:
      - visuNode            : SOFA Visu node (with 'VisualModel')
      - camera_pos (3,)
      - centroid  (3,)
      - K (3x3)             : intrinsics
      - resolution (W,H)

    Returns:
      - hit_points (M,3)    : surface points for rays that hit
      - hit_dirs   (M,3)    : corresponding world-space ray directions for the hits
    """
    fx, fy, cx, cy, res = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), tuple(int(x) for x in resolution)
    dirs_cam = get_pixel_directions(res, fx, fy, cx, cy)  # (H*W, 3)

    R = robust_look_at_rotation(np.asarray(camera_pos, dtype=np.float32),
                                np.asarray(centroid, dtype=np.float32))  # (3,3)
    dirs_world = (R @ dirs_cam.T).T.astype(np.float32)  # (N,3)
    origins = np.broadcast_to(np.asarray(camera_pos, dtype=np.float32), dirs_world.shape)  # (N,3)

    rays = np.hstack([origins, dirs_world]).astype(np.float32)  # (N,6)

    scene, _ = _build_o3d_ray_scene_from_visu(visuNode)
    rays_tensor = o3c.Tensor(rays, dtype=o3c.float32)
    out = o3d.t.geometry.RaycastingScene.cast_rays(scene, rays_tensor)

    t_hit = out['t_hit'].numpy()  # (N,)
    hit_mask = np.isfinite(t_hit)
    if not np.any(hit_mask):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    hit_points = origins[hit_mask] + dirs_world[hit_mask] * t_hit[hit_mask].reshape(-1, 1)
    hit_dirs = dirs_world[hit_mask]

    return hit_points.astype(np.float32), hit_dirs.astype(np.float32)


# -------------------------------------------------------------------------
# --- Optional: pretty figure for papers ----------------------------------
# -------------------------------------------------------------------------

def render_raycast_figure(
    legacy_mesh: o3d.geometry.TriangleMesh,
    camera_origin: np.ndarray,
    rays_oxdx: np.ndarray,          # [N,6] origins+dirs (world)
    hit_mask: np.ndarray,           # [N] bool
    t_hit: Optional[np.ndarray] = None,
    per_ray_hit_pts: Optional[np.ndarray] = None,
    outfile: str = "raycast_figure.png",
    img_size: Tuple[int, int] = (2000, 1500),
    max_rays: int = 400,
    max_hit_points: int = 4000,
    background: str = "white",
    line_width: float = 2.5,
    point_size: float = 3.0,
    zoom: float = 0.8,
    front: Optional[np.ndarray] = None,
    up: np.ndarray = np.array([0.0, 0.0, 1.0]),
    visible_window: bool = False,
) -> None:
    """
    Render a clean raycasting figure with a fixed view (for papers).
    """
    mesh = legacy_mesh  # assume already cleaned/colored if you want
    cam_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    cam_sphere.paint_uniform_color([0.85, 0.15, 0.15])  # red
    cam_sphere.translate(np.asarray(camera_origin, dtype=float))

    rays_oxdx = np.asarray(rays_oxdx, dtype=float)
    hit_mask = np.asarray(hit_mask).astype(bool).ravel()
    N = rays_oxdx.shape[0]
    stride = max(1, int(np.ceil(N / max_rays)))
    draw_idx = np.arange(0, N, stride, dtype=int)

    aabb = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(aabb.get_center())
    diag = np.linalg.norm(aabb.get_extent())
    miss_len = 0.8 * diag

    pts, lines, cols = [], [], []
    for k, i in enumerate(draw_idx):
        o = rays_oxdx[i, :3]
        d = rays_oxdx[i, 3:]
        d = d / (np.linalg.norm(d) + 1e-12)

        if hit_mask[i]:
            if t_hit is not None and np.isfinite(t_hit[i]):
                end = o + d * (float(t_hit[i]) * 0.98)
            elif per_ray_hit_pts is not None and np.all(np.isfinite(per_ray_hit_pts[i])):
                end = per_ray_hit_pts[i]
            else:
                end = o + d * miss_len
            color = [0.10, 0.65, 0.25]
        else:
            end = o + d * miss_len
            color = [0.85, 0.30, 0.30]

        lines.append([2 * k, 2 * k + 1])
        pts.extend([o, end])
        cols.append(color)

    ray_ls = o3d.geometry.LineSet()
    ray_ls.points = o3d.utility.Vector3dVector(np.asarray(pts))
    ray_ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ray_ls.colors = o3d.utility.Vector3dVector(np.asarray(cols))

    # Optionally show hit points
    hit_positions = None
    if per_ray_hit_pts is not None:
        mask = np.isfinite(per_ray_hit_pts).all(axis=1)
        hit_positions = per_ray_hit_pts[mask]
    elif t_hit is not None:
        mask = np.isfinite(t_hit)
        hit_positions = rays_oxdx[mask, :3] + (
            rays_oxdx[mask, 3:] / np.linalg.norm(rays_oxdx[mask, 3:], axis=1, keepdims=True)
        ) * t_hit[mask][:, None]

    hit_pcd = None
    if hit_positions is not None and len(hit_positions) > 0:
        if len(hit_positions) > max_hit_points:
            step = int(np.ceil(len(hit_positions) / max_hit_points))
            hit_positions = hit_positions[::step]
        hit_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hit_positions))
        hit_pcd.paint_uniform_color([0.05, 0.45, 0.15])

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=visible_window, width=img_size[0], height=img_size[1])
    vis.add_geometry(mesh)
    vis.add_geometry(cam_sphere)
    vis.add_geometry(ray_ls)
    if hit_pcd is not None:
        vis.add_geometry(hit_pcd)

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0]) if background.lower().startswith("white") else np.array([0.05, 0.05, 0.06])
    opt.light_on = True
    opt.line_width = float(line_width)
    opt.point_size = float(point_size)

    ctr = vis.get_view_control()
    if front is None:
        front = np.array([1.0, -0.7, 0.7], dtype=float)
    front = front / (np.linalg.norm(front) + 1e-12)
    ctr.set_lookat(center)
    ctr.set_front(front)
    ctr.set_up(np.asarray(up, dtype=float))
    ctr.set_zoom(float(zoom))

    # For headless export, use capture_screen_image (requires a visible GL context on some systems)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(outfile, do_render=True)
    vis.destroy_window()
    print(f"[raycast] Saved figure: {outfile}")
