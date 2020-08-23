%
% Original code taken from FDL Toolbox 
% https://github.com/LEPENDUM/FDL-Toolbox was modified to facilitate new
% processing on light field images.
% 
% -------------------------------------------------------------------------
%
% Graphical user interface for the light field rendering application
%
classdef RenderGUI < handle
    
  properties
    model

    U = []
    V = []

    Fig

    UIPanel
    viewpointPanel

    focusPanel
    radiusPanel

    imAxes
    imHandle

    ImgClicked = false;
  end
    
  methods
    function obj = RenderGUI(model, U, V)

        if(~exist('model','var') || isempty(model))
            emptyModel = RenderModel();
            obj.model = emptyModel;
            obj.createGUI([],[],[-1,1]);
        else

            obj.model = model;
            obj.model.setApShape(model.ApShapes{model.ApShapeId});

            addlistener(obj.model,'ChangeRadius', @(src,evnt)RenderGUI.changedRenderParam(obj, src, evnt));
            addlistener(obj.model,'ChangeFocus', @(src,evnt)RenderGUI.changedRenderParam(obj, src, evnt));
            addlistener(obj.model,'ChangePosition', @(src,evnt)RenderGUI.changedRenderParam(obj, src, evnt));

            if(~exist('U','var') || ~exist('V','var'))
                U = floor(size(obj.model.Ap, 1)*.5);
                V = floor(size(obj.model.Ap, 2)*.5);
                U=-U:U; 
                V=-V:V;
            end

            obj.U=U;
            obj.V=V;
            obj.createGUI(U,V);
        end

    end        

    function createGUI(obj, U, V)

        obj.Fig = figure('Color','k', 'Units', 'normalized');
        obj.Fig.Position = [0, 0.5, .3, .5];
        obj.Fig.set('WindowButtonMotionFcn',@RenderGUI.voidFcn); %tip: force update of the currentPoint property of all UI elements before triggering WindowMouseMotion event.

        %Callbacks for the main GUI
        addlistener(obj.Fig,'WindowMouseRelease', @(src,evnt)RenderGUI.mouseRelease(obj,src,evnt));


        %Main control panel
        UIPanelSize = .3;
        UIPanelColor = [.98 .98 .98]; % [0 0 0];%
        obj.UIPanel = ListLayoutPanel('Position',[.0 .0 UIPanelSize 1], 'units','normalized');
        obj.UIPanel.panel.set('BackgroundColor',UIPanelColor);

        %Image area
        obj.imAxes = axes('pos',[UIPanelSize+.01 .01 .98-UIPanelSize .98]);
        obj.imHandle = imshow(obj.model.Image);
        obj.imHandle.set('ButtonDownFcn',@(src,evnt)RenderGUI.ImgButtonDown(obj, src, evnt));

        %Add empty Component to leave space at the top of the panel.
        EmptyListLayoutComponent(obj.UIPanel, .1);

        %Add panel controling the viewpoint
        obj.viewpointPanel = ViewPointPanel(obj.UIPanel, obj.model.u0, obj.model.v0, obj.model.radius, obj.model.Ap, U, V);
        obj.viewpointPanel.setCallbackFcn(@(src,evnt)RenderGUI.moveViewPoint(obj, src, evnt));
        obj.viewpointPanel.height = .75;

        %Setup focus slider range and tick spacing
        minFocus = -2; %floor(min(D(:))*(1+1.5)/2 + max(D(:))*(1-1.5)/2);
        maxFocus = 2; %ceil(max(D(:))*(1+1.5)/2 + min(D(:))*(1-1.5)/2);
        FocusStep = 1.;
        FocusMinorStep = .125;
        FocusSlideScale = 1. / FocusMinorStep;

        %Add slider controling the focus.
        obj.focusPanel = SliderPanel( ...
          obj.UIPanel, ...
          @(src,evnt)RenderGUI.slideFocus(obj, src, evnt), ...
          'Refocus', minFocus, maxFocus, ...
          obj.model.d, ...
          FocusMinorStep, FocusStep, FocusSlideScale);
        obj.focusPanel.height = .15;

        %Setup radius slider range and tick spacing.
        radiusMinorStep = 1;
        radiusStep = 5;
        radiusSliderMax = (floor(max(numel(U), numel(V))/radiusStep)+1)*5;

        % Add slider controling the aperture radius.
        obj.radiusPanel = SliderPanel(...
          obj.UIPanel, ...
          @(src,evnt)RenderGUI.slideRadius(obj, src, evnt), ...
          'Aperture radius', 0, ...
          radiusSliderMax, obj.model.radius, ...
          radiusMinorStep,radiusStep);
        obj.radiusPanel.height = .15;                      

        %Add empty Component to leave space at the bottom of the panel.
        EmptyListLayoutComponent(obj.UIPanel, 0.1);

        %initialize the ListLayoutPanel (compute sizes of all the added components).
        obj.UIPanel.initLayout();

    end % createGUI function        

    function refreshApertureShape(obj)
        %recompute Aperture shape from modified model properties.
        obj.model.computeAperture();
        
        %display new aperture shape on screen.
        obj.viewpointPanel.setApertureShape(obj.model.Ap);
    end

    function refreshRender(obj)
%             if(obj.displayMode==0)
              %recompute image from modified modified model properties.
              obj.model.renderImage();
              %display image on screen.
              set(obj.imHandle,'Cdata', obj.model.Image);
%             elseif(obj.displayMode==1)
%               set(obj.imHandle,'Cdata', uint8(repmat( 255*(obj.model.DispMap-min(obj.model.DispMap(:))) / (max(obj.model.DispMap(:))-min(obj.model.DispMap(:))), 1,1,3)));
%             end

          pause(.02);
      end        
  end %methods
    

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Callbacks functions                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  methods ( Static )


%%%%%%%%%%%%%%%%%%%     Callbacks for model change      %%%%%%%%%%%%%%%%%%%
    function changedApertureShape(obj, src, evnt)

      switch evnt.EventName
        case 'ChangeApShape'
            obj.viewpointPanel.setRadius(src.trueRadius);
      end

      obj.refreshApertureShape();
      obj.refreshRender();

    end

    function changedRenderParam(obj, src, evnt)

      switch evnt.EventName
        case 'ChangeRadius'
          obj.radiusPanel.setValue(src.radius);
          obj.viewpointPanel.setRadius(src.radius);

        case 'ChangeFocus'
          obj.focusPanel.setValue(src.d);
          obj.model.refocusLF();

        case 'ChangePosition'
          obj.viewpointPanel.setViewPosition(src.u0, src.v0);
      end

      obj.refreshApertureShape();
      obj.refreshRender();

    end


%%%%%%%%%%%%%%%%%%%%%%%     Mouse     %%%%%%%%%%%%%%%%%%%%%%%%

    function ImgButtonDown(obj,~,~)
      obj.ImgClicked=true;
    end

    function mouseRelease(obj,~,~)
        if(obj.ImgClicked)
            obj.ImgClicked = false;
    %         if(~obj.ImgMotion)

                P = get(obj.imAxes, 'CurrentPoint');
                x = round(P(1,1));
                y = round(P(1,2));

                d = obj.model.getDisp(x,y);
                obj.model.setFocus(d);

    %         else
    %             obj.ImgMotion=false;
    %         end        
        end
    end

    function voidFcn(obj,~)
    end


%%%%%%%%%%%%%%%%%%%      Callbacks for UI elements      %%%%%%%%%%%%%%%%%%%

    function moveViewPoint(obj, src, ~)
        obj.model.setPosition( src.u0, src.v0 );
    end

    function slideRadius(obj, src, ~)
        obj.model.setRadius( src.value );
    end

    function slideFocus(obj,src,~)
    obj.model.setFocus( src.value );
    end

  end %method ( Static )
    
end %classdef
