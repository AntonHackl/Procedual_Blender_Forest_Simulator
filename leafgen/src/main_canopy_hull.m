%% Clear variables and add paths

% Clear all variables from the workspace and close all open figures
clear, close all

% Add the current directory and all the subdirectories to search path
addpath(genpath(pwd))

%% Initialize point cloud

filename = 'example-data/examplePC.mat';
ptCloud = importdata(filename);
ptCloud = double(ptCloud);

%% Initialize leaf base geometry

% Vertices of the leaf base geometry
LeafProperties.vertices = [0.0    0.0    0.0;
                           -0.04  0.02   0.0;
                           0.0    0.10   0.0;
                           0.04   0.02   0.0];

% Triangles of the leaf base geometry
LeafProperties.triangles = [1 2 3;
                            1 3 4];

%% Define target leaf distributions

% LADD relative height
TargetDistributions.dTypeLADDh = 'beta';
TargetDistributions.pLADDh = [22 3];

% LADD relative branch distance
TargetDistributions.dTypeLADDd = 'weibull';
TargetDistributions.pLADDd = [3.3 2.8];

% LADD compass direction
TargetDistributions.dTypeLADDc = 'vonmises';
TargetDistributions.pLADDc = [5/4*pi 0.1];

% LOD inclination angle
TargetDistributions.dTypeLODinc = 'dewit';
TargetDistributions.fun_pLODinc = @(h,d,c) [1 2];

% LOD azimuth angle
TargetDistributions.dTypeLODaz = 'uniform';
TargetDistributions.fun_pLODaz = @(h,d,c) [];

% LSD
TargetDistributions.dTypeLSD = 'normal';
TargetDistributions.fun_pLSD = @(h,d,c) [0.004 0.00025^2];

% Visualize the target distributions (removed for non-plotting build)

%% Define stem location

% Translate the lowest point of the cloud to the plane z=0
ptCloud = ptCloud - min(ptCloud(:,3));

% Translate the trunk of the tree to origin
tfBottom = ptCloud(:,3) < 1;
ptCloud = ptCloud - [mean(ptCloud(tfBottom,1:2)) 0];

% Set the coordinates for the stem
stemCoordinates = [   0        0                   0;
                   0.47   -0.125                11.5;
                      0        0   max(ptCloud(:,3))];

%% Set the target leaf area

totalLeafArea = 25;

%% Generate foliage inside point cloud

[Leaves,aShape] = generate_foliage_canopy_hull(ptCloud, ...
                                     TargetDistributions, ...
                                     LeafProperties,totalLeafArea, ...
                                     'StemCoordinates',stemCoordinates);

%% Visualize the foliage

% Visualization removed

% Initialize first subplot
ax1 = nexttile;

pc = aShape.Points;

% Visualization removed

% Visualization removed

% Visualization removed

% Visualization removed

% Visualization removed

% Visualization removed

% Visualization removed

% Visualization removed

% Visualization removed

%% Plot LADD marginal distributions

% Visualization removed

%% Plot LOD marginal distributions

% Visualization removed

%% Plot LSD

% Visualization removed

%% Export leaves in OBJ-format

% Precision parameter for export
precision = 5;

% Exporting to obj file
Leaves.export_geometry('OBJ',true,'leaves_export.obj',precision);