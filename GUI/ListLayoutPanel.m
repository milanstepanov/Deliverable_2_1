%
% Original code taken from FDL Toolbox 
% https://github.com/LEPENDUM/FDL-Toolbox was modified to facilitate new
% processing on light field images.
% 
% -------------------------------------------------------------------------
%
classdef ListLayoutPanel < handle
    
  properties ( SetAccess = private )
    panel
    componentList = {}  
  end    
    
  methods        
    function obj = ListLayoutPanel(varargin)
      obj.panel = uipanel(varargin{:});
      obj.panel.set('ResizeFcn', ...
        @(src,evnt)ListLayoutPanel.panelResize(obj,src,evnt));
    end

    function addComponent(obj,component)
      if(isa(component,'ListLayoutComponent'))
        alreadyContained = false;
        for i=1:length(obj.componentList)
          if(component == obj.componentList{i})
            alreadyContained = true;
            break;
          end
        end
        if(~alreadyContained)
          obj.componentList{end+1} = component;
        end
      end
    end

    function initLayout(obj)
      ListLayoutPanel.panelResize(obj,obj.panel,[]);
    end
        
  end
    
  methods( Static )        
    % Modified function to facilitate automatic resizing of the GUI
    function panelResize(obj, ~, ~)      
      % Compute the total height of created panels and normalize heights
      % based on their contribution

      totalHeight = single(0);
      for i=1:length(obj.componentList)
        totalHeight = totalHeight + obj.componentList{i}.height;
      end
      if totalHeight < 1e-10
        totalHeight = 1;
      end          

      % TODO: Incorporate margins
      cummHeight = 0;
      for i=1:length(obj.componentList)
        normHeight = obj.componentList{i}.height / totalHeight;
        cummHeight = cummHeight + normHeight;
        obj.componentList{i}.setPosition([0., 1.-cummHeight, 1., normHeight]);
      end
    end       
  end    
    
end