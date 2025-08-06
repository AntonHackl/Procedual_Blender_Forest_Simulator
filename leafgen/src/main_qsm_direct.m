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

% Visualize the target distributions
visualize_target_distributions(TargetDistributions,[0 0 0]);

%% Set the target leaf area

% totalLeafArea = 50;  % Original value - more dense foliage
totalLeafArea = 20;  % Reduced from 50 to 20 for less foliage

%% Generate foliage on QSM

[Leaves,QSMbc] = generate_foliage_qsm_direct(QSM,TargetDistributions, ...
  LeafProperties,totalLeafArea);

%% Visualize the generated foliage with the QSM

% Initialize figure
figure, clf, hold on

% Plot leaves
hLeaf = Leaves.plot_leaves();

% Set leaf color
set(hLeaf,'FaceColor',[0 150 0]./255,'EdgeColor','none');

% Plot QSM
hQSM = QSMbc.plot_model();

% Set bark color
set(hQSM,'FaceColor',[150 100 50]./255,'EdgeColor','none');

% Set figure properties
hold off;
axis equal;
xlabel('x')
ylabel('y')
zlabel('z')

%% Plot LADD marginal distributions

plot_LADD_h_QSM(QSMbc,Leaves,TargetDistributions);
plot_LADD_d_QSM(QSMbc,Leaves,TargetDistributions);
plot_LADD_c_QSM(QSMbc,Leaves,TargetDistributions);

%% Plot LOD marginal distributions

plot_LOD_inc_QSM(QSMbc,Leaves);
plot_LOD_az_QSM(QSMbc,Leaves);

%% Plot LSD

plot_LSD_QSM(QSMbc,Leaves);

%% Export leaves and QSM in OBJ-format

% Precision parameter for export
precision = 5;

% Exporting to obj files
Leaves.export_geometry('OBJ',true,'leaves_export.obj',precision);
QSMbc.export('OBJ','qsm_export.obj','Precision',precision);