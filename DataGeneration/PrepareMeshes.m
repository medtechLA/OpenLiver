clear("all");

function vol = compute_tetrahedral_volume(node, elem)
    % Computes the total volume of a tetrahedral mesh.
    num_tetrahedra = size(elem, 1);
    vol = 0;

    for i = 1:num_tetrahedra
        % Get the 4 vertices of the tetrahedron
        p1 = node(elem(i, 1), :);
        p2 = node(elem(i, 2), :);
        p3 = node(elem(i, 3), :);
        p4 = node(elem(i, 4), :);

        % Compute determinant for tetrahedral volume
        T = [p2 - p1; p3 - p1; p4 - p1];  
        tet_volume = abs(det(T)) / 6;  % Compute tetrahedral volume

        vol = vol + tet_volume;  % Sum total volume
    end
end

% Add Iso2Mesh toolbox to the MATLAB path
addpath('');

% Specify input and output folders
input_folder = '';  %labels from MID dataset
output_folder = '';
csv_output_file = fullfile(output_folder, 'mesh_statistics.csv');  

% Get all NIfTI files
nifti_files = dir(fullfile(input_folder, '*.nii.gz'));  

% Store voxel sizes to compute the median
voxel_sizes_list = [];
mesh_statistics = {};  

% **Step 1: Analyze all voxel sizes**
disp("Analyzing voxel sizes...");
for i = 1:length(nifti_files)
    nifti_filepath = fullfile(input_folder, nifti_files(i).name);
    info = niftiinfo(nifti_filepath);
    voxel_size = info.PixelDimensions;
    
    % Only store valid X and Y voxel sizes (skip Z)
    if ~all(voxel_size == 1)
        voxel_sizes_list = [voxel_sizes_list; voxel_size(1:2)];
    end
end

% Compute **median X and Y voxel sizes**
if ~isempty(voxel_sizes_list)
    median_xy = median(voxel_sizes_list, 1);
else
    median_xy = [0.7, 0.7]; % Default fallback for CT
end

% **Set default Z voxel size to 2 mm**
default_z = 2;

disp(['Median voxel size computed (XY): ', num2str(median_xy), ', Default Z = ', num2str(default_z)]);

% **Step 2: Process each NIfTI file and generate a tetrahedral mesh**
disp("Generating meshes...");
for i = 1:length(nifti_files)
    % Load the NIfTI file
    nifti_filename = nifti_files(i).name;
    nifti_filepath = fullfile(input_folder, nifti_filename);
    label_volume = niftiread(nifti_filepath);
    
    % Merge tumors into liver (if needed)
    label_volume(label_volume == 2) = 1;

    % Get voxel dimensions
    info = niftiinfo(nifti_filepath);
    voxel_size = info.PixelDimensions;

    % If voxel size is [1,1,1], replace with **median XY and Z=2**
    if all(voxel_size == 1)
        disp(['Replacing voxel size for ', nifti_filename, ' with: ', num2str(median_xy), ' ', num2str(default_z)]);
        voxel_size = [median_xy, default_z]; 
    end

    % **Downsampling Factor (Set it to 0.5 for efficiency)**
    downsampling_factor = 0.5;

    % Downsample the label volume for coarser tetrahedral meshing
    downsampled_volume = imresize3(label_volume, downsampling_factor, 'nearest');  

    % **Apply scaling correction for SOFA/3D printing**
    correction_factor = (1 / downsampling_factor)^3;  % = 8 for 0.5 downsampling

    % Generate tetrahedral mesh
    [node, elem, face] = vol2mesh(downsampled_volume > 0, ...
                                  1:size(downsampled_volume, 1), ...
                                  1:size(downsampled_volume, 2), ...
                                  1:size(downsampled_volume, 3), ...
                                  5000, 5000, 1, 'cgalmesh');

    % Scale mesh nodes to correct aspect ratio using voxel size
    node = [node(:, 1) * voxel_size(1), ...
            node(:, 2) * voxel_size(2), ...
            node(:, 3) * voxel_size(3)];

    % **Scale back mesh to real-world size**
    node = node * (1 / downsampling_factor);

    % Compute **liver volume with correction**
    liver_volume = compute_tetrahedral_volume(node, elem) * correction_factor;

    % Compute liver dimensions
    min_coords = min(node);
    max_coords = max(node);
    real_size = max_coords - min_coords;  % Width, Height, Depth in mm

    % Print the corrected real-world liver size
    disp(['Liver dimensions for ', nifti_filename, ' (mm): ', num2str(real_size), ', Volume (mm³): ', num2str(liver_volume)]);

    % Generate output file name
    [~, base_filename, ~] = fileparts(nifti_filename);
    output_filename = fullfile(output_folder, [base_filename, '_mesh.msh']);

    % Save the **real-world scaled mesh** for SOFA and 3D printing
    % Scale mesh from mm to meters before saving
    node = node * 0.001;
    savemsh(node, elem, output_filename);

    disp(['Mesh saved: ', output_filename]);

    % **Store data for CSV output**
    mesh_statistics{i, 1} = nifti_filename;
    mesh_statistics{i, 2} = size(node, 1);  % NumNodes
    mesh_statistics{i, 3} = size(face, 1);  % NumTriangles
    mesh_statistics{i, 4} = size(elem, 1);  % NumTetrahedrals
    mesh_statistics{i, 5} = real_size(1);   % Width_mm
    mesh_statistics{i, 6} = real_size(2);   % Height_mm
    mesh_statistics{i, 7} = real_size(3);   % Depth_mm
    mesh_statistics{i, 8} = liver_volume;   % Volume_mm3
end

% **Step 3: Save Data to CSV**
csv_header = {'Filename', 'NumNodes', 'NumTriangles', 'NumTetrahedrals', 'Width_mm', 'Height_mm', 'Depth_mm', 'Volume_mm3'};
csv_data = [csv_header; mesh_statistics];

% Write CSV file
disp("Saving mesh statistics to CSV...");
writecell(csv_data, csv_output_file);
disp(['Mesh statistics saved: ', csv_output_file]);

disp("All meshes processed successfully!");
