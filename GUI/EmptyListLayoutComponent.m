%
% Original code taken from FDL Toolbox 
% https://github.com/LEPENDUM/FDL-Toolbox was modified to facilitate new
% processing on light field images.
% 
% -------------------------------------------------------------------------
%
classdef EmptyListLayoutComponent < ListLayoutComponent
  
  properties ( SetAccess = private)
    panel
    
  end
  methods

    % Constructor inherited from ListLayoutComponent with parameters:
    % -listLayoutPanel
    % -height

    function obj=EmptyListLayoutComponent(Parent, height)
        obj@ListLayoutComponent(Parent); 
        if(isa(Parent,'ListLayoutPanel'))
            ParentUI = Parent.panel;
        else
            ParentUI = Parent;
        end 
        
        if exist('height', 'var')
          obj.height = single(height);
        end
        
        obj.panel = uipanel(ParentUI);
        obj.panel.Units = 'normalized';
        obj.panel.Position = [0, 0, 1, height];

    end

    % Implement abstract method from ListLayoutComponent class
    function setPosition(obj, pos)
      obj.panel.Position = pos;
    end
    
  end
end