%
% Original code taken from FDL Toolbox 
% https://github.com/LEPENDUM/FDL-Toolbox was modified to facilitate new
% processing on light field images.
% 
% -------------------------------------------------------------------------
%
% Function generating an aperture shape image.
% 
% -------------------------------------------------------------------------
%
% Inputs :
%
% - shape     A string denoting the shape of the aperture: 'dirac' or
% 'disk'.
% - apSize    An array keeping the size of the aperture.
% - apCentre  An array keeping the center of the aperture.
% - rad       The radius in number of pixels.
%
% Outputs :
%
% - Ap        Aperture shape in the pixel domain.
%

function [Ap] = buildAperture(shape, apSize, apCentre, rad)

  if nargin<3
    v0=0;
    u0=0;
    
  else
    v0 = apCentre(1);
    u0 = apCentre(2);
    
  end

  szApX = apSize.szApX;
  szApY = apSize.szApY;  
  
  if(~exist('rad', 'var'))
    rad = 0;
  end

switch shape

  case 'dirac'
    Ap = zeros(szApY,szApX);
    Ap(v0, u0)=1;
  
  case 'disk'
    Ap = insertShape(zeros(szApY,szApX), ...
                     'FilledCircle', [u0 v0 rad], ...
                     'Color','white','LineWidth',1,'Opacity',1);
    Ap = Ap(:,:,1);
  
  otherwise
    Ap = zeros(szApY,szApX);
    Ap(v0, u0)=1;

end
