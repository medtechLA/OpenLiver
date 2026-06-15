import os
import gmsh
import pyvista as pv
import open3d as o3d
import numpy as np


def convert_pyvista_to_open3d(pv_mesh):
    """Convert a PyVista PolyData mesh to an Open3D TriangleMesh."""
    points = np.asarray(pv_mesh.points)
    faces = np.asarray(pv_mesh.faces.reshape(-1, 4))[:, 1:]  # PyVista encodes number of points per face
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(points)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh

def process_mesh(input_msh_file, output_vtk_file, output_obj_file,output_stl_file, smoothing_iterations=5):
    # Step 1: Initialize GMSH and open the .msh file
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)  # Enable terminal output for debugging
    gmsh.open(input_msh_file)

    # Extract the surface mesh (generate 2D surface mesh from 3D mesh)
    gmsh.model.mesh.generate(2)  # Generate the 2D surface mesh

    # Save the surface mesh in .vtk format
    temp_vtk_file = ""
    gmsh.write(temp_vtk_file)
    print(f"Surface mesh written to {temp_vtk_file}")

    # Load the surface mesh using PyVista
    mesh = pv.read(temp_vtk_file)

    # Extract the surface (convert UnstructuredGrid to PolyData)
    surface = mesh.extract_surface()


    o3d_mesh = convert_pyvista_to_open3d(surface)
    pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=30000)
    pcd.estimate_normals()
    print("Running Poisson surface reconstruction...")
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    poisson_mesh.compute_vertex_normals()


    o3d.io.write_triangle_mesh( os.path.join(output_folder, f"{ output_obj_file}"), poisson_mesh)
    o3d.io.write_triangle_mesh(os.path.join(output_folder, f"{ output_stl_file}"), poisson_mesh)
    print(f"Poisson reconstructed mesh saved to {output_obj_file}")
    # Final step: Finalize and close GMSH
    gmsh.finalize()


def process_all_meshes(input_folder, output_folder, smoothing_iterations=5):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all .msh files in the input folder
    msh_files = [f for f in os.listdir(input_folder) if f.endswith('.msh')]

    # Loop through each .msh file
    for msh_file in msh_files:
        input_msh_file = os.path.join(input_folder, msh_file)

        # Generate output filenames (remove .msh extension and add new extensions)
        base_filename = os.path.splitext(msh_file)[0]
        output_vtk_file = os.path.join(output_folder, f"{base_filename}.vtk")
        output_obj_file = os.path.join(output_folder, f"{base_filename}.obj")
        output_stl_file = os.path.join(output_folder, f"{base_filename}.stl")

        # Process the current mesh
        print(f"Processing {msh_file}...")
        process_mesh(input_msh_file, output_vtk_file, output_obj_file,output_stl_file, smoothing_iterations)


# Example usage
input_folder = r''  # Folder containing your .msh files
output_folder = r''  # Folder where you want to save the .obj and .vtk files

process_all_meshes(input_folder, output_folder)
