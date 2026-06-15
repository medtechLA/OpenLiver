# 🩺 OpenLiver Simulation Pipeline

---

## 📘 Overview

**OpenLiver Simulation (Data Generation)** is a physics-based data generation pipeline using the [SOFA framework](https://www.sofa-framework.org/) for realistic liver deformation.

The pipeline generates camera-consistent intraoperative partial observations by raycasting deformed liver meshes with calibrated virtual camera intrinsics. Two sensor setups are supported:

* **Microsoft HoloLens 2 Long Throw**
* **Intel RealSense D435**

The generated data can be used for liver registration, correspondence learning, deformation analysis, and training pipelines such as LiverMatch.

---

## 🧩 1. Data Generation — OpenLiver Simulation

### 🗂️ Architecture

```text
DataGeneration/
├── config.yaml          # All parameters: paths, sensors, forces, simulation settings
├── main.py              # Runs the full simulation pipeline
├── sofa_scene.py        # SOFA scene setup and controller logic
├── raycast.py           # Open3D-based raycasting
├── io_utils.py          # Paths, saving, checkpoints, and metadata
├── prepare_meshes.m     # MATLAB script for generating meshes from MSD labels
└── msh2obj.py           # Converts .msh meshes to .obj format for SOFA
```

Each liver simulation applies a unique biomechanical deformation, captures multi-view partial observations, and saves per-frame `.npz` datasets compatible with the LiverMatch training scripts.

---

## 🧠 OpenLiver Dataset Summary

| Category            | Description                                                                                                                                        |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Physics**         | Nearly incompressible liver parenchyma modeled as a tetrahedral FEM with Young’s modulus between 2–5 kPa and Poisson’s ratio 0.35.                 |
| **Forces**          | Two bottom regions are fixed, while two additional random regions receive ramped forces with random direction and magnitude over 50 frames.        |
| **Sensors**         | Two virtual cameras are supported: Microsoft HoloLens 2 Long Throw at 320×288 resolution and Intel RealSense D435 at 640×480 resolution.           |
| **Raycasting**      | Partial views are generated using Open3D raycasting while respecting camera intrinsics, camera distance, and view angle.                           |
| **Camera Settings** | Camera distances of 0.5 m and 1.0 m are used, with viewing angles from −60° to +60°.                                                               |
| **Noise Model**     | Gaussian noise is added along camera rays with σ ∈ [0, 5 mm] to emulate stochastic depth uncertainty.                                              |
| **Outputs**         | Full and partial point clouds, deformation flows, correspondences, TRE probes, volumetric mesh data, and deformation metadata are saved per frame. |

---

## 🧱 Mesh Preparation from MSD Labels

To reproduce the OpenLiver dataset, liver meshes must first be generated from the liver segmentation labels of the Medical Segmentation Decathlon dataset.

The repository includes two preprocessing scripts for this step:

| Script             | Purpose                                                                                              |
| ------------------ | ---------------------------------------------------------------------------------------------------- |
| `prepare_meshes.m` | Generates liver surface/volume meshes from MSD liver segmentation labels.                            |
| `msh2obj.py`       | Converts the generated `.msh` mesh files into `.obj` format for use in the SOFA simulation pipeline. |

The preprocessing workflow is:

1. **Generate meshes from MSD labels**

   Run the MATLAB script:

   ```bash
   prepare_meshes.m
   ```

   This script processes the MSD liver segmentation labels and creates the mesh files required for simulation.

2. **Convert `.msh` files to `.obj`**

   Convert the generated mesh files using:

   ```bash
   python msh2obj.py
   ```

   The resulting `.obj` files are used as input liver geometries in the SOFA-based simulation.

3. **Update mesh paths**

   After mesh generation and conversion, update the corresponding input paths in:

   ```bash
   DataGeneration/config.yaml
   ```

   Make sure the configuration points to the generated liver mesh files.

---

## 🧱 Directory Structure

Each liver is simulated with multiple random force configurations and camera setups.

```text
<output_base>/
├── Liver_001/
│   ├── Simulation_01/
│   │   ├── Camera_angle0_radius0p50/
│   │   │   ├── HL/              # HoloLens 2 frames
│   │   │   └── RS/              # RealSense frames
│   │   ├── metadata.json
│   │   └── _checkpoints/
│   ├── Simulation_02/
│   │   └── ...
│   └── ...
├── Liver_002/
│   └── ...
└── ...
```

Each camera folder contains per-frame `.npz` files generated during the simulation.

---

## 📦 Saved `.npz` File Contents

Each `frame_XXXX.npz` file contains the following data fields:

| Key                          | Description                                                                            |
| ---------------------------- | -------------------------------------------------------------------------------------- |
| `full_pc_undeformed`         | Complete undeformed liver surface point cloud.                                         |
| `full_pc_deformed`           | Complete deformed liver surface point cloud.                                           |
| `partial_pc_undeformed`      | Partial undeformed surface generated from the camera view.                             |
| `partial_pc_deformed`        | Partial deformed surface generated by raycasting.                                      |
| `partial_pc_deformed_noisy`  | Noisy deformed partial point cloud with Gaussian ray-direction noise.                  |
| `flow_full`                  | Full deformation flow field between undeformed and deformed liver surface points.      |
| `step_flow`                  | Incremental deformation flow for the current frame.                                    |
| `corr_pairs_deformed`        | Correspondences between deformed partial and deformed full point clouds.               |
| `corr_pairs_undeformed`      | Correspondences between partial and full point clouds in the undeformed configuration. |
| `visibility_ratio_surface_*` | View-dependent visible surface ratio for the current camera setup.                     |
| `tetrahedra_indices`         | Tetrahedral mesh connectivity.                                                         |
| `tetrahedra_vertices_*`      | Undeformed and deformed tetrahedral mesh vertices.                                     |
| `tre_points_undeformed`      | Undeformed TRE probe points.                                                           |
| `tre_points_deformed`        | Deformed TRE probe points.                                                             |
| `force_node_indices`         | Indices of nodes receiving external forces.                                            |

---

## ⚙️ Configuration

All simulation parameters are controlled through a single YAML configuration file:

```text
DataGeneration/config.yaml
```

The configuration includes:

* input mesh paths
* output paths
* SOFA simulation parameters
* biomechanical material parameters
* force magnitudes and force regions
* fixed regions
* camera angles and distances
* HoloLens 2 and RealSense camera intrinsics
* frame settings
* noise settings
* checkpoint settings

Running the simulation with the original provided configuration reproduces the dataset generation settings used in the OpenLiver paper.

---

## ▶️ Running the Simulation

### 1. Install SOFA Framework

Install the SOFA framework by following the official instructions:

```text
https://sofa-framework.github.io/doc/getting-started/build
```

### 2. Prepare Liver Meshes

Generate liver meshes from the MSD segmentation labels:

```bash
prepare_meshes.m
```

Then convert the generated `.msh` files to `.obj` format:

```bash
python msh2obj.py
```

### 3. Configure Paths

Update the input and output paths in:

```bash
DataGeneration/config.yaml
```

Make sure the configuration points to the generated `.obj` liver meshes.

### 4. Run the Simulation

Run the full data generation pipeline with:

```bash
python main.py --config DataGeneration/config.yaml
```

### 5. Resumable Execution

Each simulation automatically saves checkpoints under:

```text
_checkpoints/
```

The simulation can be interrupted and restarted safely. When restarted, the pipeline resumes from the last saved frame.

### 6. Output

The generated datasets are saved under the output directory specified in the configuration file.

---

## 📂 OpenLiver Dataset Reproduction

To reproduce the exact OpenLiver dataset used in the paper:

1. Download the liver segmentation data from the Medical Segmentation Decathlon dataset.
2. Generate the liver meshes using `prepare_meshes.m`.
3. Convert the generated `.msh` files to `.obj` using `msh2obj.py`.
4. Set the mesh and output paths in `DataGeneration/config.yaml`.
5. Use the same simulation parameters, camera angles, camera distances, sensor intrinsics, noise levels, force settings, and frame settings as provided in the configuration file.
6. Run the simulation using:

   ```bash
   python main.py --config DataGeneration/config.yaml
   ```

Using the original configuration allows the OpenLiver dataset to be recreated with the same simulation and acquisition settings used in the paper.

---

## 📌 Notes

* The simulation uses SOFA for biomechanical deformation.
* Open3D is used for raycasting and generation of camera-consistent partial point clouds.
* HoloLens 2 and RealSense camera models are simulated using calibrated camera intrinsics.
* The generated `.npz` files include full and partial point clouds, deformation flows, correspondences, TRE probes, and volumetric mesh data.
* Checkpointing allows long simulation runs to be resumed after interruption.

---

## 📖 Citation

If you use this code or dataset in your research, please cite our paper.

```bibtex
@article{openliver,
  title   = {OpenLiver},
  author  = {Paper information will be added after publication},
  journal = {To appear},
  year    = {2026}
}
```

The final paper citation and link will be added after publication.
