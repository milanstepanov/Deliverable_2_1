%
% Original code taken from FDL Toolbox 
% https://github.com/LEPENDUM/FDL-Toolbox was modified to facilitate new
% processing on light field images.
% 
% -------------------------------------------------------------------------
%
classdef ListLayoutComponent < handle
    
  properties
    height = 1
  end
    
  methods

    function obj = ListLayoutComponent(listLayoutPanel, height)

      if(exist('height','var') && isscalar(height) && isnumeric(height))
        obj.height = height;
      end

      if(exist('listLayoutPanel','var'))
        obj.addToListLayout(listLayoutPanel);
      end

    end
    function addToListLayout(obj, listLayoutPanel)
    if(isa(listLayoutPanel,'ListLayoutPanel'))
        listLayoutPanel.addComponent(obj);
    end
  end

  end
    
  methods( Abstract )
    setPosition(obj, pos)
  end
    
end