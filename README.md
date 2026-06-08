# 🩺 OpenLiver Simulation Pipeline


---

## 📘 Overview


**OpenLiver Simulation (Data Generation)**  
   A physics-based pipeline using the [SOFA framework](https://www.sofa-framework.org/) for realistic liver deformation.  
   It performs raycasting with calibrated camera intrinsics (HoloLens 2 & RealSense D435) to generate camera-consistent partial observations.

---

## 🧩 1. Data Generation — OpenLiver Simulation

### 🗂️ Architecture

DataGeneration/

default.yaml ← All parameters (paths, sensors, forces, etc.)

main.py ← runs the full simulation

sofa_scene.py ← SOFA scene setup & controller logic

raycast.py ← Raycasting with Open3D 

io_utils.py ← Paths, saving, checkpoints, metadata



Each liver simulation applies a unique biomechanical deformation, captures multi-view partial observations, and saves per-frame `.npz` datasets compatible with the LiverMatch training scripts.

---

### 🧠 OpenLiver Dataset  Summary

| Category | Description |
|-----------|--------------|
| **Physics** | Nearly incompressible parenchyma modeled as a tetrahedral FEM (Young’s modulus 2–5 kPa, Poisson’s ratio 0.35). |
| **Forces** | Two bottom regions fixed; two other random regions receive ramped forces (random direction/magnitude) over 50 frames. |
| **Sensors** | Two virtual cameras:<br>• *Microsoft HoloLens 2 Long Throw* (320×288)<br>• *Intel RealSense D435* (640×480) |
| **Raycasting** | Partial views generated via Open3D raycasting, respecting intrinsics, distance (0.5 m / 1.0 m), and angles (−60° → +60°). |
| **Noise Model** | Gaussian noise added along camera rays (σ ∈ [0, 5 mm]) to emulate realistic depth uncertainty. |
| **Outputs** | Full/partial point clouds, flows, correspondences, TRE probes, and volumetric deformation metadata per frame. |

---

### 🧱 Directory Structure

Each liver is simulated with multiple random force configurations and camera setups:

<output_base>/<br>
├─ Liver_001/<br>
│ ├─ Simulation_01/<br>
│ │ ├─ Camera_angle0_radius0p50/<br>
│ │ │ ├─ HL/ ← HoloLens2 frames<br>
│ │ │ └─ RS/ ← RealSense frames<br>
│ │ ├─ metadata.json<br>
│ │ └─ _checkpoints/<br>
│ └─ Simulation_02/<br>
│ ...<br>
├─ Liver_002/<br>
│ ...<br>



Each `frame_XXXX.npz` file includes:

| Key | Description |
|-----|--------------|
| `full_pc_undeformed`, `full_pc_deformed` | Complete liver surfaces before/after deformation. |
| `partial_pc_undeformed`, `partial_pc_deformed` | Partial surfaces from raycasting. |
| `partial_pc_deformed_noisy` | Same as above, but with realistic Gaussian noise. |
| `flow_full`, `step_flow` | Full and incremental deformation flow fields. |
| `corr_pairs_deformed`, `corr_pairs_undeformed` | Partial-to-full correspondences. |
| `visibility_ratio_surface_*` | Fraction of visible surface vertices per view. |
| `tetrahedra_indices`, `tetrahedra_vertices_*` | Volumetric mesh data. |
| `tre_points_undeformed`, `tre_points_deformed` | Centroids for TRE computation. |
| `force_node_indices` | Nodes receiving external forces. |

---

### ⚙️ Configuration (`configs.yaml`)

All parameters are summarized in one YAML file.
[DataGeneration/config.yaml](configs.yaml)

running the simulation with the original provided configuration would replicate the dataset used in our paper.<br>



### ▶️ Running the Simulation 


1. **Install SOFA Framework**<br>
Follow instructions at  https://sofa-framework.github.io/doc/getting-started/build.

2. **run the simulation**<br>
   python main.py --config configs/default.yaml

3. **Resumable excecution**<br>
   Each simulation automatically saves checkpoints under _checkpoints/.<br>
   You can safely interrupt and restart; progress resumes at the last saved frame.<br>
4. **Output**<br>
   The generated datasets will be saved under the specified output directory.
   
### OpenLiver Dataset 
To reproduce the exact openLiver dataset in our paper, use the Liver segmentation data from the Medical Segmentation Decathlon and set the paths in the configuration file accordingly.
Make sure to use the exact same parameters for forces, camera angles, noise levels, and simulation settings as specified in the `config.yaml` to ensure consistency with the published dataset.

### Citation
If you use this code or dataset in your research, please cite our paper:
paper will be linked after publication.
