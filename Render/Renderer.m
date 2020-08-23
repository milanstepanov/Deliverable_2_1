%
% Original code taken from FDL Toolbox 
% https://github.com/LEPENDUM/FDL-Toolbox was modified to facilitate new
% processing on light field images.
% 
% -------------------------------------------------------------------------
%
% Main class for light field rendering and interaction from synthesized
% views.
% 
% -------------------------------------------------------------------------
% 
% The Renderer object constructor has the following arguments:
%
% - LF          Light field image
% - Disparities Disparity map of each perspective view.
% - isLinear    A flag employed to apply gamma correction after rendering.
%
% The methods available in the Renderer class are:
%
% - obj.renderImage()
%		Render the image with the current Renderer's parameters (e.g. focus,
% aperture parameters).
%
% - obj.refocusLF()
%   Refocus LF to a new focal plane by shifting views for a specified
%   offset. 
%
% - obj.computeAperture()
%		Updates the internal representation of the aperture with the current
% aperture parameters: center of the aperture and radius.
%
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% - Set functions
%
% - obj.setRadius(radius)
%		Set the aperture radius of the image to render. The radius depends on
%	the angular size of the input LF image. It can be access via the
%	variable obj.radius.
%
% - obj.setFocus(focus)
%		Set the focus parameter of the image to render. The focus is defined in
% the form of the disparity. 0th disparity corresponds to the current focal 
% plane while changing the disparity the light field is rendered to a new 
% focal plane. The focus parameter can be access via opt.d variable.
%
% - obj.setPosition(u,v)
%		Set the center of the aperture. The center can be access via variables
% obj.u0 and obj.v0.
%
% - obj.setApShape(ApShapeName)
%   Set the shape of aperture by name. Currently, supported shapes are
% "dirac" and "disk".
%
% - obj.setApShapeId(ApShapeId)
%   Set the shape of aperture by shape ID. Currently, supported shapes are
%   "dirac" and "disk" are coded with 1 and 2 respectively.
%
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% - Get functions
%
% - obj.getDisp(x, y)
%   Get current disparity (focal parameter).
%
% - obj.getCenterMatCoords()
%   Get current aperture centre in matrix coordinates.
%
%--------------------------------------------------------------------------
%
% See also RenderAppMain

classdef Renderer < Observable
% Note: inherits from the custom class Observable instead of the built-in
% handle class to overcome a bug in the event notification (in some cases
% the notification does not produce any effect).
    
    properties ( Access = private )
      
        apSize
        
        AngularSize % angular size of LF
        SpatialSize % spatial size of LF
        LF   
        LF_working
        
        DispMaps
        
        Image_
                
    end
    
    properties ( Access = public, Constant )
        ApShapes={'dirac','disk'}
        numApShapes = length(Renderer.ApShapes);
    end
    
    properties ( SetAccess = private)
        ApShapeId = 2
        radius = single(2)

        d = single(0)
        
        % define the centre of the aperture
        u0 = 0
        v0 = 0
                
        Ap
        Image=0
    end
    
    properties
        isLinear = false
        gammaOffset = 0
    end
    
    methods
        function obj = Renderer(LF, Disparities, isLinear)
            
            if(nargin==0)
                return;
            end
                        
            fprintf('Initializating data...\n');
            
            obj.AngularSize = size(LF, 1);
            obj.SpatialSize = [size(LF, 3), size(LF, 4)];
            
            if isempty(isLinear)
              obj.isLinear = false;
            else
              obj.isLinear = isLinear;
            end
            
            s = size(LF);
            
            if isempty(Disparities)
              obj.DispMaps = zeros(s(1),s(2),1,s(4),s(5));
              disp("Disparity map not available.");
            else
              obj.DispMaps = Disparities;
            end
              
            
            obj.LF = LF;
            obj.LF_working = LF;            
            
            obj.apSize.szApY = size(obj.LF, 1);
            obj.apSize.szApX = size(obj.LF, 2);
                       
            %Initialize Aperture data
            obj.computeAperture();
            
            %Intialize Image
            obj.renderImage();
            
            fprintf('Initialization done.\n');
            
        end
            
        %Image render function
        function renderImage(obj)
          
            obj.Image_ = squeeze(sum(obj.LF_working.*obj.Ap, [1,2])) ./ ...
                         sum(obj.Ap, 'all');
            if obj.isLinear
              obj.Image_ = obj.BT709_gamma(obj.Image_);
            end
            obj.Image = obj.Image_;
            
        end
        
        % Refocus LF
        function refocusLF(obj)
          
          if abs(obj.d) < 1e-10
            obj.LF_working = obj.LF;
            
          else            
            % shift LF code from LFToolbox
            TVSlope = double(obj.d);
            SUSlope = double(obj.d);

            v = linspace(1,size(obj.LF, 3), size(obj.LF, 3));
            u = linspace(1,size(obj.LF, 4), size(obj.LF, 4));

            VOffsetVec = linspace(-0.5,0.5, size(obj.LF, 1)) * ...
                         TVSlope*(size(obj.LF, 1)-1);
            UOffsetVec = linspace(-0.5,0.5, size(obj.LF, 2)) * ...
                         SUSlope*(size(obj.LF, 2)-1);

            obj.LF_working = zeros(size(obj.LF), 'like', obj.LF);
            for TIdx = 1:size(obj.LF, 1) 
              VOffset = VOffsetVec(TIdx);

              for SIdx = 1:size(obj.LF, 2) 
                UOffset = UOffsetVec(SIdx);
                CurSlice = squeeze(obj.LF(TIdx, SIdx, :,:, :));

                Interpolant = griddedInterpolant( CurSlice );

                CurSlice = Interpolant( {v+VOffset, u+UOffset, 1:size(obj.LF,5)} );

                obj.LF_working(TIdx,SIdx, :,:, :) = CurSlice;
              end
              
            end
            
            
          end
        end
        
        %Aperture update function
        function computeAperture(obj)
          
          centerCords = obj.getCenterMatCoords();
          [obj.Ap] = buildAperture(Renderer.ApShapes{obj.ApShapeId}, ...
                                   obj.apSize, ...
                                   centerCords, ...
                                   obj.radius);
        end
        
        %Set methods
        function setRadius(obj,radius)
            if(radius ~= obj.radius)
                obj.radius = max(0,radius);
                notify(obj,'ChangeRadius');
            end
        end
        
        function setFocus(obj,d)
            if(d ~= obj.d)
                obj.d = d;
                notify(obj,'ChangeFocus');
            end
        end
        
        function setPosition(obj,u0,v0)
            if(u0 ~= obj.u0 || v0 ~= obj.v0)
                obj.u0 = u0;
                obj.v0 = v0;
                notify(obj,'ChangePosition');
            end
        end
                
        function setApShape(obj, ApShapeName)
            id = find(strcmp(ApShapeName,Renderer.ApShapes));
            if(~isempty(id))
                if(obj.ApShapeId ~= id)
                    obj.ApShapeId = id;
                    notify(obj,'ChangeApShape');
                end
            else
                warning(['unknown aperture shape (', ApShapeName, ')'] );
            end
            
        end
        
        function setApShapeId(obj, ApShapeId)
            if(ApShapeId>0 && ApShapeId <= Renderer.numApShapes && round(ApShapeId)==ApShapeId && obj.ApShapeId ~= ApShapeId)
                obj.ApShapeId = ApShapeId;
                notify(obj,'ChangeApShape');
            end
            
        end
        
        %Get methods
        function d = getDisp(obj, x, y)
          
          cords = obj.getCenterMatCoords();
          
          if ~obj.imgPixels([y,x])
            d = obj.d;
            
          else          
            d = 0;
            if ~isempty(obj.DispMaps)
              d = obj.DispMaps(cords(1), cords(2), y, x, 1);
            end
            
          end
          
        end
        
        function cords = getCenterMatCoords(obj)
          
          % aperture is a matrix with the y axis in direction up-down
          v0_ = ceil(obj.apSize.szApY*.5) - obj.v0;
          u0_ = ceil(obj.apSize.szApX*.5) + obj.u0;
          
          cords = [v0_, u0_];
          
        end
        
        function in_bound = imgPixels(obj, cords)
          in_bound = true;
          height = obj.SpatialSize(1);
          width  = obj.SpatialSize(2);
          
          if cords(1) > height || cords(1) < 1 || cords(2) > width || cords(2) < 1
            in_bound = false;
          end
        end
        
    end %methods
    
    methods ( Static )
        %Apply BT709 standard gamma correction to convert from linear RGB data to BT709.
        %(negative input values are clipped to 0).
        function Iout = BT709_gamma(Iin)
            Mask = Iin < 0.018;
            Iout = max(0,Iin * 4.5) .* Mask + (1.099*max(Iin,0).^0.45 - 0.099) .* (~Mask);
        end
        
    end %methods ( Static )
    
    
end
