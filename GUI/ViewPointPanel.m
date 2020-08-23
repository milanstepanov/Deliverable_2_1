%
% Original code taken from FDL Toolbox 
% https://github.com/LEPENDUM/FDL-Toolbox was modified to facilitate new
% processing on light field images.
% 
% -------------------------------------------------------------------------
%
classdef ViewPointPanel < ListLayoutComponent
    
    
  properties
    PosColor = [0 .3 0]
    ApColor = [.1 .7 .1]
    ApAlpha = .5
  end

  properties ( SetAccess = private )
    u0=0
    v0=0
    radius=0
    Ap=[]
    
    main_panel
    text
    panel
    
    plotAxes
    clickAxes 
    ApAxes
    ApImg
    ApPos

    uPosEditor
    uText
    vPosEditor
    vText

    U = []
    V = []
    UVLim
    offset = 1    
    uPlotCenter = 0
    vPlotCenter = 0
    hasUVcoords = false

    callbackFcn = []

  end

  properties ( Access = private )
    buttonClicked = false
    
    ValboxDecimalDigits=2
  end


  methods

    function obj = ViewPointPanel(Parent, u0, v0, radius, Ap, U, V, UVLim)
          obj@ListLayoutComponent(Parent);
          if(isa(Parent,'ListLayoutPanel'))
              ParentUI = Parent.panel;
          else
              ParentUI = Parent;
          end  

          if(exist('u0','var') && exist('v0','var'))
              obj.u0 = u0;
              obj.v0 = v0;
          end
          if(exist('radius','var'))
              obj.radius = radius;
          end
          if(exist('Ap','var'))
            obj.Ap = Ap;
          end

          %Set optional U,V coordinates of intitial viewpoints
          if(exist('U','var') && exist('V','var'))
              obj.U = U; obj.V = V;
          end
          obj.hasUVcoords = ~isempty(obj.U) && ~isempty(obj.V);
          if(numel(obj.U) ~= numel(obj.V))
              warning(strcat("U and V lists of initial view ", ...
                             "positions have different sizes. ", ...
                             "They will be ignored."));
              obj.hasUVcoords = false;
              obj.U = []; obj.V = [];
          end

          %Set limits U,V coordinates of the panel
          if(exist('UVLim','var'))
              obj.UVLim = UVLim;
              obj.uPlotCenter = mean(obj.UVLim(1:2));
              obj.vPlotCenter = mean(obj.UVLim(3:4));
              
          elseif(obj.hasUVcoords)            
              obj.uPlotCenter = (max(obj.U(:))+min(obj.U(:)))/2;
              obj.vPlotCenter = (max(obj.V(:))+min(obj.V(:)))/2;
              obj.UVLim(1:2) = [obj.uPlotCenter - (obj.uPlotCenter-min(obj.U(:))), ...
                                obj.uPlotCenter + (max(obj.U(:))-obj.uPlotCenter)];
              obj.UVLim(3:4) = [obj.vPlotCenter - (obj.vPlotCenter-min(obj.V(:))), ...
                                obj.vPlotCenter + (max(obj.V(:))-obj.vPlotCenter)];
          end

          %Main viewpoint panel
          obj.main_panel = uipanel('Parent', ParentUI, ...
                                   'Units','normalized');
          
          text_sep = .9;
          obj.text = uicontrol('Parent', obj.main_panel, ...
                               'Style','text', ...
                               'String', "Aperture center");
          obj.text.HorizontalAlignment = 'center';
          obj.text.FontUnits = 'normalized';
          obj.text.FontSize = .3;
          obj.text.Units = 'normalized';
          obj.text.Position = [0. text_sep 1. 1.-text_sep];
        
        
          obj.panel = uipanel('Parent', obj.main_panel, ...
                              'Units','normalized');
          obj.panel.Position = [0. 0. 1. text_sep];
          obj.panel.set('ResizeFcn', ...
            @(src,evnt)ViewPointPanel.panelResize(obj,src,evnt));

          %Axes of the viewpoints
          obj.plotAxes = axes('pos',[0 0 1 1], 'Parent',obj.panel);
          line([0,0], [intmin intmax], ...
               'Color','red', 'LineStyle',':', 'Parent',obj.plotAxes);
          line([intmin intmax], [0 0], ...
               'Color','red', 'LineStyle',':', 'Parent',obj.plotAxes);
          
          obj.plotAxes.set('XLim', obj.UVLim(1:2)+[-obj.offset obj.offset], ...
                           'YLim', obj.UVLim(3:4)+[-obj.offset obj.offset]);
          obj.plotAxes.set('XTickMode','manual','YTickMode','manual', ...
                           'XColor',[.5 .5 .5], 'YColor',[.5 .5 .5], ...
                           'XGrid','on', 'YGrid','on', 'TickLength', [0 0]);
          obj.plotAxes.set('XTick', ...
            round(obj.UVLim(1)-obj.offset):round(obj.UVLim(2)+obj.offset));
          obj.plotAxes.set('YTick', ...
            round(obj.UVLim(3)-obj.offset):round(obj.UVLim(4)+obj.offset));
          
          obj.plotAxes.XColor = 'none';
          obj.plotAxes.YColor = 'none';
          obj.plotAxes.Color = [1, 1, 1];
          

          %Axes of the aperture image
          obj.ApAxes = axes('Parent', obj.panel);
          nPointsX = numel(obj.UVLim(1):obj.UVLim(2))+1;
          nPointsY = numel(obj.UVLim(3):obj.UVLim(4))+1;
          obj.ApAxes.Position = [0.5/nPointsX, ...
                                 0.5/nPointsY, ...
                                 (nPointsX-1)/nPointsX, ...
                                 (nPointsY-1)/nPointsY];
          obj.ApAxes.XLim = [0, numel(obj.UVLim(1):obj.UVLim(2))] + .5;
          obj.ApAxes.YLim = [0, numel(obj.UVLim(3):obj.UVLim(4))] + .5;

          hold on;

          %Initialize Aperture image
          ap = bsxfun(@times, ...
                      ones([size(obj.Ap),3]), ...
                      permute(obj.ApColor,[3 1 2]));
          obj.ApImg = image(ap, 'Parent', obj.ApAxes);
          obj.ApImg.set('AlphaData',max(0,(obj.Ap*obj.ApAlpha)));

          %Marker of the view position
          xCentre = obj.UVLim(2)+1;
          yCentre = obj.UVLim(4)+1;
          obj.ApPos = line(xCentre,yCentre, ...
                           'Marker','+','MarkerSize',10, ...
                           'Color',obj.PosColor,'LineWidth',1.5, ...
                           'Parent', obj.ApAxes);
          obj.ApAxes.Color = 'none';
          obj.ApAxes.XColor = 'none';
          obj.ApAxes.YColor = 'none';
          obj.ApAxes.YTickLabel = [];
          obj.ApAxes.XTickLabel = [];

          hold off

          %Text box editors for u and v positions
          box_height = .03;
          box_width = .1;
          obj.uText = uicontrol('Parent', obj.panel, ...
                                'Style','text','String','u=', ...
                                'Units','normalized', ...
                                'Position',[.01,box_height+.01,box_width,box_height], ...
                                'BackgroundColor','white', ...
                                'FontUnits','normalized', ...
                                'FontSize',1);
          
          obj.vText = uicontrol('Parent', obj.panel, ...
                                'Style','text','String','v=', ...
                                'Units','normalized', ...
                                'Position',[.01,.01,box_width,box_height], ...
                                'BackgroundColor','white', ...
                                'FontUnits','normalized', ...
                                'FontSize',1);
          

          % Invisible axes on top of other elements only used to receive
          % mouse clicks.
          clickAxes = axes('Parent', obj.panel, ...
                           'Color', 'none', ...
                           'XColor','none', 'YColor','none', ...
                           'Position',[0 0 1 1], 'Visible','on');
          clickAxes.set('ButtonDownFcn', ...
                        @(src,evnt)ViewPointPanel.mouseClick(obj,src,evnt));
                      
                      
          obj.uPosEditor = uicontrol('Parent',obj.panel, ...
                                     'Style','edit', ...
                                     'Units','normalized', ...
                                     'Position',[box_width+.01,box_height+.01,box_width,box_height]);
          set(obj.uPosEditor, ...
            'Callback', @(src,evnt) ViewPointPanel.editUVPos(obj,src,evnt));
          
          obj.vPosEditor = uicontrol('Parent',obj.panel, ...
                                     'Style','edit', ...
                                     'Units','normalized', ...
                                     'Position',[box_width+.01,.01,box_width,box_height]);
          set(obj.vPosEditor, ...
            'Callback', @(src,evnt) ViewPointPanel.editUVPos(obj,src,evnt));
          
          obj.setViewPosition(obj.u0, obj.v0);

      end

%%%%%%%%%%%%%%%     Methods updating the panel display     %%%%%%%%%%%%%%%%

      function updateApertureShape(obj)

          obj.ApPos.set('XData',obj.UVLim(2)+1 + obj.u0, ...
                        'YData',obj.UVLim(4)+1 + obj.v0); % this op repeats

          % updating Cdata which demands flipping y axis
          ap = flipud(obj.Ap);
          obj.ApImg.set('Cdata', ...
                        bsxfun(@times, ...
                               ones([size(ap),3]), ...
                               permute(obj.ApColor,[3 1 2])));
          obj.ApImg.set('AlphaData',max(0,(ap*obj.ApAlpha)));

      end

      function centerView(obj)
          if(obj.hasUVcoords)
              obj.scale = 3;
              obj.uPlotCenter = (max(obj.U(:))+min(obj.U(:)))/2;%mean(U(:));%
              obj.vPlotCenter = (max(obj.V(:))+min(obj.V(:)))/2;%mean(V(:));%
              obj.UVLim(1:2) = [obj.uPlotCenter - obj.scale*(obj.uPlotCenter-min(obj.U(:))), obj.uPlotCenter + obj.scale*(max(obj.U(:))-obj.uPlotCenter)];
              obj.UVLim(3:4) = [obj.vPlotCenter - obj.scale*(obj.vPlotCenter-min(obj.V(:))), obj.vPlotCenter + obj.scale*(max(obj.V(:))-obj.vPlotCenter)];
          else
              obj.uPlotCenter = 0;
              obj.vPlotCenter = 0;
          end
          obj.UpdatePlotLimits(1);
      end

      function UpdatePlotLimits(obj)
        obj.plotAxes.set('XLim', obj.UVLim(1:2)+[-obj.offset obj.offset]);
        obj.plotAxes.set('YLim', obj.UVLim(3:4)+[-obj.offset obj.offset]);

      end


%%%%%%%%%%%%%%%%%%%%%%%%%     Access methods     %%%%%%%%%%%%%%%%%%%%%%%%%%

      function setViewPosition(obj,u0,v0)
          u0 = round(u0);
          v0 = round(v0);

          if u0<obj.UVLim(1), u0 = obj.UVLim(1); end
          if u0>obj.UVLim(2), u0 = obj.UVLim(2); end
          if v0<obj.UVLim(3), v0 = obj.UVLim(3); end
          if v0>obj.UVLim(4), v0 = obj.UVLim(4); end

          obj.u0 = u0;
          obj.v0 = v0;

          xCentre = obj.UVLim(2)+1;
          yCentre = obj.UVLim(4)+1;

          obj.ApPos.XData = xCentre + u0;
          obj.ApPos.YData = yCentre + v0; % plot so (0,0) is located at the
                                          % bottom left
          obj.uPosEditor.set('String',num2str(u0));
          obj.vPosEditor.set('String',num2str(v0));
          pause(.00001); 
      end

      function setRadius(obj, radius)
          obj.radius = radius;
      end

      function setApertureShape(obj, Ap)
          obj.Ap = Ap;
          obj.updateApertureShape();
      end

      function setCallbackFcn(obj, callbackFcn)
          if(isa(callbackFcn, 'function_handle'))
              obj.callbackFcn = callbackFcn;
          else
              obj.callbackFcn=[];
          end
      end

      % Implement abstract method from ListLayoutComponent class
      function setPosition(obj,pos)
          obj.panel.Position = pos;
          obj.updateApertureShape();
      end


  end % methods



%%%%%%%%%%%%%%%%%%%%%%%     Callback functions     %%%%%%%%%%%%%%%%%%%%%%%%

  methods ( Static )

    function panelResize(obj,~,~)
        obj.UpdatePlotLimits();
    end

    function mouseClick(obj,~,~)
        
      P = get(obj.plotAxes, 'CurrentPoint');        
      obj.buttonClicked = true;

      obj.setViewPosition(P(1,1), P(1,2));
      if(~isempty(obj.callbackFcn))
        obj.callbackFcn(obj,'MovePosition');
      end

    end

    function editUVPos(obj,src,~)
          val = str2double(src.get('String'));
          if(isnan(val))
              if(src == obj.uPosEditor)
                  src.set('String',num2str(round(obj.u0,obj.ValboxDecimalDigits)));
              else
                  src.set('String',num2str(round(obj.v0,obj.ValboxDecimalDigits)));
              end
          else
              if(src == obj.uPosEditor)
                  obj.setViewPosition( val, obj.v0 );
                  uicontrol(obj.vPosEditor);%set the focus to the other box.
              else
                  obj.setViewPosition( obj.u0, val );
                  uicontrol(obj.uPosEditor);%set the focus to the other box.
              end
              if(~isempty(obj.callbackFcn)), obj.callbackFcn(obj,'MovePosition');end
          end
      end

  end % methods ( Static )

end