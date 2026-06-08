
"""
sofa_scene.py — Liver simulation scene setup for OpenLiver Simulation

Contains:
  • create_scene – builds SOFA scene graph for one liver
  • ForceCycleController – applies ramped forces over time
  • select_regions_6parts – divides mesh into regions for fixing/forcing
"""

import Sofa
import Sofa.Core
import numpy as np
import random


# -------------------------------------------------------------------------
# --- REGION SELECTION ----------------------------------------------------
# -------------------------------------------------------------------------
def select_regions_6parts(mechanical_object):
    """
    Divide liver mesh into 6 parts (left/middle/right × top/bottom).
    Select 2 bottom parts as fixed, and 2 others as force regions.
    """
    positions = np.array([list(pos) for pos in mechanical_object.position.value])
    min_x, min_y, min_z = positions.min(axis=0)
    max_x, max_y, max_z = positions.max(axis=0)

    # Split along X (left/mid/right)
    x_third1 = min_x + (max_x - min_x) / 3
    x_third2 = min_x + 2 * (max_x - min_x) / 3

    regions = {
        "left_top": [],
        "left_bottom": [],
        "middle_top": [],
        "middle_bottom": [],
        "right_top": [],
        "right_bottom": []
    }

    for idx, (x, y, z) in enumerate(positions):
        if x < x_third1:
            part = "left_"
        elif x < x_third2:
            part = "middle_"
        else:
            part = "right_"

        vertical = "top" if z > (min_z + (max_z - min_z) / 2) else "bottom"
        regions[part + vertical].append(idx)

    # pick fixed and force regions
    bottom_parts = [k for k in regions.keys() if "bottom" in k]
    fixed_parts = random.sample(bottom_parts, 2)
    valid_force_candidates = [k for k in regions if k not in fixed_parts and len(regions[k]) > 0]
    force_parts = random.sample(valid_force_candidates, 2)

    fixed_nodes = []
    for part in fixed_parts:
        fixed_nodes.extend(regions[part])

    force_nodes = {part: regions[part] for part in force_parts}

    print(f"[Region Selection] Fixed parts: {fixed_parts}, Force parts: {force_parts}")
    return fixed_parts, force_parts, fixed_nodes, force_nodes


# -------------------------------------------------------------------------
# --- FORCE CONTROLLER ----------------------------------------------------
# -------------------------------------------------------------------------
class ForceCycleController(Sofa.Core.Controller):
    """
    Applies ramped external forces to preselected node regions over simulation frames.
    """

    def __init__(self, liverNode, force_regions, force_cycle_length=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.liverNode = liverNode
        self.force_regions = force_regions
        self.force_cycle_length = force_cycle_length
        self.frame_count = 0

        self.force_profiles = {}
        self.max_magnitudes = {}
        self.fixed_parts = None
        self.force_parts = None
        self.fixed_nodes = None
        self.force_nodes = None

        self.initialize_forces()

    def initialize_forces(self):
        """
        Initialize a random direction and magnitude per region.
        """
        for region in self.force_regions:
            direction = np.random.uniform(-1, 1, size=3)
            n = np.linalg.norm(direction)
            direction = direction / n if n > 0 else np.array([0, 0, 1], dtype=float)
            magnitude = random.uniform(1, 5)  # [N] (unitless scaling)
            self.force_profiles[region] = direction
            self.max_magnitudes[region] = magnitude
            print(f"[Init] Force profile for {region}: dir={direction}, max={magnitude:.2f}")

    def ramp_factor(self, frame):
        """
        Linear ramp up to maximum at `force_cycle_length` frames.
        """
        return min(frame / (self.force_cycle_length - 1), 1.0)

    def onAnimateBeginEvent(self, event):
        """
        Called before each animation step. Updates the externalForce field.
        """
        mechanical_object = self.liverNode.getObject('dofs')
        forces = [[0.0, 0.0, 0.0] for _ in range(len(mechanical_object.position.value))]

        for region_name, region_nodes in self.force_regions.items():
            factor = self.ramp_factor(self.frame_count)
            force_vec = self.force_profiles[region_name] * self.max_magnitudes[region_name] * factor
            force_per_node = (force_vec / len(region_nodes)).tolist()
            for idx in region_nodes:
                forces[idx] = force_per_node

        mechanical_object.externalForce.value = forces
        self.frame_count += 1


# -------------------------------------------------------------------------
# --- SCENE CREATION ------------------------------------------------------
# -------------------------------------------------------------------------
def create_scene(rootNode, liver_mesh_path, liver_surface_path, cfg):
    """
    Build a SOFA scene for one liver simulation.
    Returns (liverNode, visuNode, topo, controller)
    """

    rootNode.gravity = [0, 0, 0]
    rootNode.dt = cfg["run"]["dt"]
    rootNode.addObject('DefaultAnimationLoop')

    # Load required SOFA plugins
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.IO.Mesh')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mapping.Linear')
    rootNode.addObject('RequiredPlugin', name='Sofa.GL.Component.Rendering3D')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.FEM.Elastic')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.StateContainer')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mass')

    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Dynamic')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.ODESolver.Backward')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Algorithm')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Intersection')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Geometry')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Projective')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Iterative')
    rootNode.addObject('RequiredPlugin', name='SofaPython3')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.IO.Mesh')

    # --- Liver Node ---
    liverNode = rootNode.addChild('Liver')
    liverNode.addObject('EulerImplicitSolver',
                        rayleighStiffness=cfg["physics"]["rayleigh_stiffness"],
                        rayleighMass=cfg["physics"]["rayleigh_mass"])
    liverNode.addObject('CGLinearSolver', iterations=cfg["run"]["frames"], tolerance=1e-12, threshold=1e-12)

    liverNode.addObject('MeshGmshLoader', name='meshLoader', filename=liver_mesh_path)
    liverNode.addObject('TetrahedronSetTopologyContainer', name='topo', src='@meshLoader')
    liverNode.addObject('MechanicalObject', template='Vec3', name='dofs', position='@meshLoader.position')
    liverNode.addObject('DiagonalMass', template='Vec3,Vec3', totalMass=cfg["physics"]["total_mass"])
    liverNode.addObject('TetrahedronSetGeometryAlgorithms', template='Vec3')
    liverNode.addObject('TetrahedralCorotationalFEMForceField',
                        name="FEM", template="Vec3",
                        poissonRatio=cfg["physics"]["poisson_ratio"],
                        youngModulus=cfg["physics"]["young_modulus"])

    # --- Boundary & Force Region Selection ---
    fixed_parts, force_parts, fixed_nodes, force_regions = select_regions_6parts(liverNode.getObject('dofs'))
    liverNode.addObject('FixedProjectiveConstraint', template='Vec3', indices=fixed_nodes)

    # --- Controller for Forces ---
    controller = ForceCycleController(
        liverNode,
        force_regions=force_regions,
        force_cycle_length=cfg["forces"]["cycle_length"]
    )
    controller.fixed_parts = fixed_parts
    controller.force_parts = force_parts
    controller.fixed_nodes = fixed_nodes
    controller.force_nodes = force_regions
    liverNode.addObject(controller)

    # --- Visual Model ---
    visuNode = liverNode.addChild('Visu')
    visuNode.addObject('MeshOBJLoader', name='meshLoader_0', filename=liver_surface_path)
    visuNode.addObject('OglModel', name='VisualModel', src='@meshLoader_0')
    visuNode.addObject('BarycentricMapping', input='@../dofs', output='@VisualModel')

    print(f"[Scene] Liver scene created with Young's modulus = {cfg['physics']['young_modulus']}")
    return liverNode, visuNode, liverNode.getObject('topo'), controller
