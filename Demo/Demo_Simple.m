%% Run Rendering Application

%% Clear and prepare environment
clear; close all; clc;

py.sys.setdlopenflags(int32(10));

os = py.importlib.import_module('os');
clear os.environ.MKL_NUM_THREADS

%% PARAMETERS TO SET
% Set path to the python with installed PyTorch module
python_path = [];
root_dir = []; % keeps folders "Demo", "GUI", "Render" and Synthesis"
if isempty(root_dir) % if the folder is not defined it is assumed that the
                     % current folder is the root
  root_dir = pwd;
end

path_to_test_lenslet = fullfile(root_dir,"Demo/IMG_2312_eslf.png");
angul = 7; % set the angular size of the input lenslet
is_8bits = false;

gamma = .4;

%% Set Python
% If MATLAB is invoked from the terminal with activated appropriate
% these step is not necessary.
if ~isempty(python_path)
  try
    pyversion(python_path); 
  end
end

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
new_model = true;
net = mod.OAVS(...
  fullfile(pwd, "model_corr_b3_no_corners_no_decay_grad_flowers_gamma_best"),...
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
  scale  = 1.;
  offset = 0.;
  if new_model
    scale=2.;
    offset=.5;
  end
  sample.c1 = extractView(LF, 1,1, 7, scale,offset);
  sample.c2 = extractView(LF, pq_max,1, 7, scale,offset);
  sample.c3 = extractView(LF, 1,pq_max, 7, scale,offset);
  sample.c4 = extractView(LF, pq_max,pq_max, 7, scale,offset);

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
function view = extractView(LF, p, q, angul, scale, offset)
% extractView  Extract a view from light field (LF) image.
%   LF      -   Input light field image with assumed dimension format HWC.
%   p       -   Horizontal angular position in LF.
%   q       -   Vertical angular position in LF.
%   angul   -   Angular size of the input LF.
%   scale   -   Value to scale input intensities.
%   offset  -   Value to add to the scaled intensities
%               i = scale*i - offset
%
%   return  - A specified view LF(p,q,:,:,:) in a dimension format 1CHW.

  if nargin<5
    offset = 0.;
    scale  = 1.;
  end

  [h,w,c] = size(LF);
  h_ = h/angul;
  w_ = w/angul;
  
  view = zeros(1,c,h_,w_, 'single');
  
  for c_id=1:c
    view(:,c_id,:,:) = ...
      (single(LF(p:angul:end,q:angul:end,c_id))/255.) * scale - offset;
  end
  
end

function img = normalize(img, offset, scale)
  if nargin<2
    offset=0.;
    scale =1.;
  end
  img = (img+offset) * scale;
end


