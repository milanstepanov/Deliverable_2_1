%% Run Rendering Application

%% Clear environment
clear; close all; clc;

%% PARAMETERS TO SET
% Set path to the python with installed PyTorch module
python_path = [];
root_dir = []; % keeps folders "Demo", "GUI", "Render" and Synthesis"

path_to_test_lenslet = fullfile(root_dir,"Demo/IMG_2312_eslf.png");
angul = 7; % set the angular size of the input lenslet
is_8bits = true;

gamma = .45;

%% Set Python
% If MATLAB is invoked from the terminal with activated appropriate
% these step is not necessary.
if ~isempty(python_path)
  try
    pyversion(python_path); 
  end
end

% flag = int32(bitor(2, 8));
% py.sys.setdlopenflags(flag);

%%
if isempty(root_dir)
  root_dir = pwd;
end

%%
addpath(fullfile(root_dir, 'Demo'), ...
        fullfile(root_dir, 'GUI'), ...
        fullfile(root_dir, 'Render'), ...
        fullfile(root_dir, 'Synthesis'));
      
%% Synthesis
  % Temporary moving to Synthesis folder to generate LF
  cd(fullfile(root_dir, 'Synthesis'));

  %% Load python libraries
  mod = py.importlib.import_module('oavs');
  py.importlib.reload(mod);

  %% Load network
  net = mod.OAVS(...
    fullfile(pwd, ...
             "/model_corr_b3_no_corners_no_decay_grad_best"), ...
    false);

  %% Load Light Field data
  path_to_lenslet = path_to_test_lenslet;
  LF = imread(path_to_lenslet);

  %%
  % clip to 8bits
  if ~is_8bits
    LF = uint8(bitshift(LF, -8));
  end


  %%
  if gamma>0
    LF = uint8(((single(LF)./255).^gamma).*255);
  end

  %% Extraction of the corner views from the loaded lenset image

  step=1.; % Corresponds to the baseline between neighbouring views
           % allows to synthesize LF more densely 
  pq_min = 1 + floor((angul-7)/2);         
  pq_max = pq_min + 7 - 1; % TODO: select the position of the corner views

  % extract corner views
  sample.c1 = extractView(LF, pq_min,pq_min, angul);
  sample.c2 = extractView(LF, pq_max,pq_min, angul);
  sample.c3 = extractView(LF, pq_min,pq_max, angul);
  sample.c4 = extractView(LF, pq_max,pq_max, angul);

  %%
  a = numel(1:step:7); % angular resolution of the synthesized LF
  [h, w, c] = size(LF);
  LF_prime = zeros(round(h/angul), round(w/angul), c, a, a, 'single'); % buffer to store synthesized LF
  Disparities = zeros(round(h/angul), round(w/angul), 1, a, a, 'single'); % buffer to store disparities 

  tic;
  for p=1:step:7
    for q=1:step:7

      % angular positions of the view to be synthesized
      sample.p = p-1; % 0-indexed notion used in Python
      sample.q = q-1;

      % network inference
      prediction = net.forward(sample);    

      % network outputs
      pred = squeeze(uint8(prediction{'pred'})); % synthesized view
      D = squeeze(single(prediction{'disp'})); % disparities
      m = squeeze(single(prediction{'m'})); % fusion matrix
      clear prediction;    

      % generate disparity map
      [M,I] = max(m, [], 3);
      I_ = medfilt2(I);
      I_(I_==0) = I(I_==0);    
      disp_map = zeros([size(I_), 4]);
      for i=1:4
        disp_map(:,:,i) = (I_==i);
      end    
      d = sum(D.*disp_map, 3);

      LF_prime(:,:,:, (1/step)*(q-1)+1,(1/step)*(p-1)+1) = single(pred)/255;
      Disparities(:,:,:,(1/step)*(q-1)+1,(1/step)*(p-1)+1) = d;

    end
  end
  elapsed_time = toc;
  disp(strcat("Time per view: ", num2str(elapsed_time/a/a)));


%% Start Rendering Application
cd(root_dir)
RenderAppMain(LF_prime, Disparities);

%% Auxiliary
function view = extractView(LF, p, q, angul)
% extractView  Extract a view from light field (LF) image.
%   LF      -   Input light field image with assumed dimension format HWC.
%   p       -   Horizontal angular position in LF.
%   q       -   Vertical angular position in LF.
%   angul   -   Angular size of the input LF.
%
%   return  - A specified view LF(p,q,:,:,:) in a dimension format 1CHW.

  [h,w,c] = size(LF);
  h_ = h/angul;
  w_ = w/angul;
  
  view = zeros(1,c,h_,w_, 'single');
  
  for c_id=1:c
    view(:,c_id,:,:) = single(LF(p:angul:end,q:angul:end,c_id))/255.;
  end
  
end

