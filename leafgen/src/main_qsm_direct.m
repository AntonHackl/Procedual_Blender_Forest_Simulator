%% Clear variables and add paths

% Clear all variables from the workspace and close all open figures
clear, close all

% Add the current directory and all the subdirectories to search path
addpath(genpath(pwd))

%% Initialize QSM

filename = "example-data/generated_tree.mat";
QSM = importdata(filename);

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

%% Define target leaf distributions

% LADD relative height - modified for more concentration at top while maintaining some spread
% Using beta distribution with parameters that concentrate foliage more at the top
TargetDistributions.dTypeLADDh = 'beta';
% TargetDistributions.pLADDh = [22 3];  % Original parameters - very concentrated at top
% TargetDistributions.pLADDh = [2 2];  % Previous parameters - uniform distribution
TargetDistributions.pLADDh = [8 3];  % Modified to concentrate more at top while keeping some spread

% LADD relative branch distance - modified to concentrate foliage on the outside
TargetDistributions.dTypeLADDd = 'weibull';
% TargetDistributions.pLADDd = [3.3 2.8];  % Original parameters - more uniform distribution
TargetDistributions.pLADDd = [2.0 1.5];  % Modified to concentrate foliage more on the outside of branches

% LADD compass direction
TargetDistributions.dTypeLADDc = 'vonmises';
TargetDistributions.pLADDc = [5/4*pi 0.1];

% LOD inclination angle
TargetDistributions.dTypeLODinc = 'dewit';
TargetDistributions.fun_pLODinc = @(h,d,c) [1 2];

% LOD azimuth angle
TargetDistributions.dTypeLODaz = 'uniform';
TargetDistributions.fun_pLODaz = @(h,d,c) [];

% LSD - restored to original leaf size
TargetDistributions.dTypeLSD = 'normal';
TargetDistributions.fun_pLSD = @(h,d,c) [0.008 0.00025^2];  % Restored to original parameters - larger leaves
% TargetDistributions.fun_pLSD = @(h,d,c) [0.002 0.0001^2];  % Previous reduced size

% Visualization removed

%% Set the target leaf area

% totalLeafArea = 50;  % Original value - more dense foliage
totalLeafArea = 20;  % Reduced from 50 to 20 for less foliage

%% Generate foliage on QSM

[Leaves,QSMbc] = generate_foliage_qsm_direct(QSM,TargetDistributions, ...
  LeafProperties,totalLeafArea);

%% Visualize the generated foliage with the QSM

% Visualization removed

%% Plot LADD marginal distributions

% Visualization removed

%% Plot LOD marginal distributions

% Visualization removed

%% Plot LSD

% Visualization removed

%% Export leaves and QSM in OBJ-format

% Precision parameter for export
precision = 5;

% Exporting to obj files
Leaves.export_geometry('OBJ',true,'leaves_export.obj',precision);
QSMbc.export('OBJ','qsm_export.obj','Precision',precision);