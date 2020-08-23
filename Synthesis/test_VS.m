%% Test View Synthesis approach
archs = py.importlib.import_module('network_v2');
%%
torch = py.importlib.import_module('torch');

%%
net = archs.OcclusionAwareVS(7, 4, 0);

%%
checkpoint = torch.load("model_corr_b3_no_corners_no_decay_grad_best", ...
                        torch.device('cpu'));
          %%
          net.load_state_dict(checkpoint{'model_state_dict'});
%%
net.eval();
%%
sample.p=3;sample.q=3;
%%
pred = net.forward(sample.p, sample.q, sample.c1, sample.c2, sample.c3, sample.c4)



%% Clear environment
clear; close all; clc;

%% Set path to the python with installed PyTorch module
pyversion /home/milan/anaconda3/envs/tfc/bin/python

%% Set the path to the root folder
root_dir = '/media/milan/SAVE/Code/MATLAB/deliverable';
cd(fullfile(root_dir, 'Synthesis'));

%% Load python libraries
mod = py.importlib.import_module('oavs');
py.importlib.reload(mod);

% Load network
net = mod.OAVS(fullfile(pwd, "/model_corr_b3_no_corners_no_decay_grad_best"));

%% Load lenslet LF image
path_to_lenslet = strcat("/home/milan/tmp_training_data/data/", ...
                         "test/Cars.png");
LF = imread(path_to_lenslet);
%%
gamma = .45;
if gamma>0
  LF = uint8(((single(LF)./255).^gamma).*255);
end

%% Extraction of the corner views from the loaded lenset image

angul = 7; % TODO: set the angular size of the input lenslet

% evaluate the performance of the view synthesis method
to_compute_psnr = true; % TODO: set to "true" to visualize
if to_compute_psnr  
  psnrs = zeros(angul,angul);
  
end

% visualize synthesized views in parallel with original ones
to_show=true; % TODO: set to "true" to visualize
to_show_disp=true; % TODO: set to "true" to visualize disparities of
                   % each syntheized view
                   % Note: works only in to_show mode.
if to_show
  or_fig = figure("Name", "Original");
  vs_fig = figure("Name", "Synthesized");
  
  if to_show_disp
    disp_fig = figure("Name", "Disparities");
  end
  
else
  to_show_disp=false;
  
end 
  
%
% Note: Pressing "o" (followed by ENTER) will stop visualization and
% contunue synthesizing LF.
%


step=1.; % if evaluating the performance the step must be set to 1
         % Corresponds to the baseline between neighbouring views
         % allows to synthesize LF more densely 
         

pq_max = 7; % TODO: select the position of the corner views

% extract corner views
sample.c1 = extractView(LF, 1,1, 7);
sample.c2 = extractView(LF, pq_max,1, 7);
sample.c3 = extractView(LF, 1,pq_max, 7);
sample.c4 = extractView(LF, pq_max,pq_max, 7);

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
    
    if to_compute_psnr
      q_ = round((q-1)/6*(pq_max-1))+1;
      p_ = round((p-1)/6*(pq_max-1))+1;
      
      gt = LF(q_:7:end, p_:7:end, :);
      
      mse = mean((single(pred)/255 - single(gt)/255).^2, 'all');
      psnr = -10*log10(mse);
      psnrs(p,q) = psnr;
    end
    
    LF_prime(:,:,:, (1/step)*(q-1)+1,(1/step)*(p-1)+1) = single(pred)/255;
    Disparities(:,:,:,(1/step)*(q-1)+1,(1/step)*(p-1)+1) = d;
    
    if to_show
      figure(or_fig);
      imshow(gt);
      title(strcat("(", num2str(p), ", ", num2str(q), ")")) 
      figure(or_fig);
      or_fig.WindowState = 'maximized';
      pause(0.01);
      
      figure(vs_fig);
      imshow(pred);
      title(strcat("PSNR: ", num2str(round(psnr))));
      figure(vs_fig);
      vs_fig.WindowState = 'maximized';
      pause(0.01);
      
      if to_show_disp
        figure(disp_fig);
        showDisparities(D, disp_fig);
      end      
      
      key = input("Any:", 's');
      if strcmp(key, 'o')
        to_show = false;
      end
      
    end
    
  end
end
elapsed_time = toc;
disp(strcat("Time per view: ", num2str(elapsed_time/a)));

%%
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

function showDisparities(D, disp_fig)
% SHOWDISPARITIES Visualize the disparity maps. It is assumed disparities
% lie in the range [-4,4].

  figure(disp_fig);
  subplot(2,2,1);
  title('c1');
  imshow((squeeze(D(:,:,1))*.25+1)*.5);
  subplot(2,2,3);
  title('c2');
  imshow((squeeze(D(:,:,2))*.25+1)*.5);
  subplot(2,2,2);
  title('c3');
  imshow((squeeze(D(:,:,3))*.25+1)*.5);
  subplot(2,2,4);
  title('c4');
  imshow((squeeze(D(:,:,4))*.25+1)*.5);
  disp_fig.WindowState = 'maximized';
  
  pause(0.01);
  
end

function img = prepareData(im)
% PREPAREDATA Scale input image intensity range to [0,1] range and reshape
% the input structure from HWC to 1CHW.
  [h,w,c,~] = size(im);
  img = zeros(1,c,h,w, 'single');
  for c_id=1:c
    img(:,c_id,:,:) = single(im(:,:,c_id))/255.;
  end
end

function reloadPy()
% RELOADPY  Reload python modules.
%           During debugging it is usefull to reload py modules after
%           making modifications.

  warning('off','MATLAB:ClassInstanceExists')
  
  mod = py.importlib.import_module('oavs');
  py.importlib.reload(mod);
  
  net_mod = py.importlib.import_module('network_v2');
  py.importlib.reload(net_mod);

end

