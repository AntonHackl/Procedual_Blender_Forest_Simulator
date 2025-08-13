function run_leaf_generation_with_params(leaf_params)
%% Leaf Generation with Parameters from Python
% This function accepts parameters from Python and generates foliage accordingly

% Clear all variables from the workspace and close all open figures
% But preserve the leaf_params variable passed from Python
clear QSM Leaves QSMbc TargetDistributions LeafProperties totalLeafArea precision hLeaf hQSM
% close all - DISABLED FOR BLENDER INTEGRATION

% Add the current directory and all the subdirectories to search path
addpath(genpath(pwd))

%% Initialize QSM

% Resolve path to generated_tree.mat relative to this file
script_dir = fileparts(mfilename('fullpath'));
filename = fullfile(script_dir, 'example-data', 'generated_tree.mat');
QSM = importdata(filename);
% Unwrap if saved as struct with field 'qsm'
if isstruct(QSM) && isfield(QSM, 'qsm')
    QSM = QSM.qsm;
end

%% Initialize leaf base geometry

% Vertices of the leaf base geometry
LeafProperties.vertices = [0.0    0.0    0.0;
  -0.04  0.02   0.0;
  0.0    0.10   0.0;
  0.04   0.02   0.0];

% Triangles of the leaf base geometry
LeafProperties.triangles = [1 2 3;
  1 3 4];

%% Define petiole length sampling interval

LeafProperties.petioleLengthLimits = [0.08 0.10];

%% Define target leaf distributions using parameters from Python

% LADD relative height - using parameters from Python
TargetDistributions.dTypeLADDh = 'beta';
if isfield(leaf_params, 'pLADDh')
    % Convert cell array to numeric array if needed and ensure double type
    if iscell(leaf_params.pLADDh)
        pLADDh_array = double(cell2mat(leaf_params.pLADDh));
    else
        pLADDh_array = double(leaf_params.pLADDh);
    end
    fprintf('pLADDh: %f %f\n', pLADDh_array(1), pLADDh_array(2));
    TargetDistributions.pLADDh = pLADDh_array;
else
    % Default fallback - modified for more concentration at top while maintaining some spread
    % Original: TargetDistributions.pLADDh = [22 3];  % Very concentrated at top
    % Previous: TargetDistributions.pLADDh = [2 2];  % Uniform distribution
    TargetDistributions.pLADDh = [8 3];  % Current: Moderate top concentration
end

% LADD relative branch distance - using parameters from Python
TargetDistributions.dTypeLADDd = 'weibull';
if isfield(leaf_params, 'pLADDd')
    % Convert cell array to numeric array if needed and ensure double type
    if iscell(leaf_params.pLADDd)
        pLADDd_array = double(cell2mat(leaf_params.pLADDd));
    else
        pLADDd_array = double(leaf_params.pLADDd);
    end
    fprintf('pLADDd: %f %f\n', pLADDd_array(1), pLADDd_array(2));
    TargetDistributions.pLADDd = pLADDd_array;
else
    % Default fallback - modified to concentrate foliage on the outside
    % Original: TargetDistributions.pLADDd = [3.3 2.8];  % More uniform distribution
    TargetDistributions.pLADDd = [2.0 1.5];  % Current: Concentrated on outside
end

% LADD compass direction
TargetDistributions.dTypeLADDc = 'vonmises';
TargetDistributions.pLADDc = [5/4*pi 0.1];

% LOD inclination angle
TargetDistributions.dTypeLODinc = 'dewit';
TargetDistributions.fun_pLODinc = @(h,d,c) [1 2];

% LOD azimuth angle
TargetDistributions.dTypeLODaz = 'uniform';
TargetDistributions.fun_pLODaz = @(h,d,c) [];

% LSD - using parameters from Python
TargetDistributions.dTypeLSD = 'normal';
if isfield(leaf_params, 'fun_pLSD')
    % Convert cell array to numeric array if needed and ensure double type
    if iscell(leaf_params.fun_pLSD)
        fun_pLSD_array = double(cell2mat(leaf_params.fun_pLSD));
    else
        fun_pLSD_array = double(leaf_params.fun_pLSD);
    end
    fprintf('fun_pLSD: %f %f\n', fun_pLSD_array(1), fun_pLSD_array(2));
    TargetDistributions.fun_pLSD = @(h,d,c) fun_pLSD_array;
else
    % Default fallback - restored to original leaf size
    % Original: TargetDistributions.fun_pLSD = @(h,d,c) [0.004 0.00025^2];  % Original size
    % Previous: TargetDistributions.fun_pLSD = @(h,d,c) [0.002 0.0001^2];  % Reduced size
    TargetDistributions.fun_pLSD = @(h,d,c) [0.008 0.00025^2];  % Current: Larger leaves
end

% Visualization removed
% visualize_target_distributions(TargetDistributions,[0 0 0]);

%% Set the target leaf area using parameter from Python

if isfield(leaf_params, 'totalLeafArea')
    % Convert to numeric if needed and ensure double type
    if iscell(leaf_params.totalLeafArea)
        totalLeafArea = double(cell2mat(leaf_params.totalLeafArea));
    else
        totalLeafArea = double(leaf_params.totalLeafArea);
    end
    fprintf('totalLeafArea: %f\n', totalLeafArea);
else
    % Default fallback - reduced for less dense foliage
    % Original: totalLeafArea = 50;  % More dense foliage
    totalLeafArea = 20;  % Current: Reduced for less foliage
end

%% Generate foliage on QSM

[Leaves,QSMbc] = generate_foliage_qsm_direct(QSM,TargetDistributions, ...
  LeafProperties,totalLeafArea);

%% Visualize the generated foliage with the QSM - DISABLED FOR BLENDER INTEGRATION

% Visualization removed
% figure, clf, hold on

% Plot leaves
% hLeaf = Leaves.plot_leaves();

% Set leaf color
% set(hLeaf,'FaceColor',[0 150 0]./255,'EdgeColor','none');

% Plot QSM
% hQSM = QSMbc.plot_model();

% Set bark color
% set(hQSM,'FaceColor',[150 100 50]./255,'EdgeColor','none');

% Visualization removed

%% Plot LADD marginal distributions - DISABLED FOR BLENDER INTEGRATION

% Visualization removed

%% Plot LOD marginal distributions - DISABLED FOR BLENDER INTEGRATION

% Visualization removed

%% Plot LSD - DISABLED FOR BLENDER INTEGRATION

% Visualization removed

%% Export leaves and QSM in OBJ-format

% Precision parameter for export
precision = 5;

% Get the current script directory and create export paths
script_dir = fileparts(mfilename('fullpath'));
export_dir = fullfile(script_dir, 'example-data');

% Ensure the export directory exists
if ~exist(export_dir, 'dir')
    mkdir(export_dir);
end

% Create full paths for export files
leaves_export_path = fullfile(export_dir, 'leaves_export.obj');
qsm_export_path = fullfile(export_dir, 'qsm_export.obj');

% Exporting to obj files
Leaves.export_geometry('OBJ', true, leaves_export_path, precision);
QSMbc.export('OBJ', qsm_export_path, 'Precision', precision);

fprintf('Exported leaves to: %s\n', leaves_export_path);
fprintf('Exported QSM to: %s\n', qsm_export_path);

end 