%
% Original code taken from FDL Toolbox 
% https://github.com/LEPENDUM/FDL-Toolbox was modified to facilitate new
% processing on light field images.
% 
% -------------------------------------------------------------------------
%
% Main rendering application :
% Create the graphical user interface.
%
%--------------------------------------------------------------------------
%
% Function Call:
%
% RenderAppMain(LF, Disparities, isLinear)
% - LF           Input light field. Stored in format HWCVU.
% - Disparities  Disparity map of every LF view. The disparity maps are
% used to focus LF at different focal
% - isLinear     A flag denoting if the input LF is represented in linear
% color space or not. If 'isLinear' set to 'true' rendered view will be
% gamma corrected.
%
%--------------------------------------------------------------------------
%
% GUI Features:
% - Change viewpoint (from the viewpoint panel).
% - Change aperture radius (use slider).
% - Change focus (click on the image or use the slider).
%
%--------------------------------------------------------------------------
%

function gui = RenderAppMain(LF, Disparities, isLinear)

    LF = permute(LF, [4,5,1,2,3]);

    if(~exist('Disparities','var'))
        Disparities = [];
    else
        Disparities = permute(Disparities, [4,5,1,2,3]);
    end
    
    if(~exist('isLinear','var'))
      isLinear=[];
    end

    renderer = Renderer(LF, Disparities, isLinear);
    gui = RenderGUI(renderer);
    
    renderer.setPosition(gui.viewpointPanel.uPlotCenter, ...
                         gui.viewpointPanel.vPlotCenter);
    
end