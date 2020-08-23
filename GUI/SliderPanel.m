%
% Original code taken from FDL Toolbox 
% https://github.com/LEPENDUM/FDL-Toolbox was modified to facilitate new
% processing on light field images.
% 
% -------------------------------------------------------------------------
%
classdef SliderPanel < ListLayoutComponent
    
  properties ( SetAccess = private )
    callbackFcn=[]
    value
  end
    
  properties ( Access = private )

    slideScale = 1
    ValboxDecimalDigits = 2;       

    panel
    text
    slider
    sliderHG
    ValEditor

    skipJSliderCallback = false

  end
    
    
    methods
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%     Constructor     %%%%%%%%%%%%%%%%%%%%%%%%%%%
      function obj = SliderPanel(Parent, callbackFcn, name, ...
                                 min, max, val, ...
                                 stepMinor, stepMajor, slideScale, ...
                                 Position)
        obj@ListLayoutComponent(Parent);
        if(isa(Parent,'ListLayoutPanel'))
          ParentUI = Parent.panel;
        else
          ParentUI = Parent;
        end            

        ParentColor = get(ParentUI,'BackgroundColor');

        %Main matlab uipanel
        obj.panel = uipanel('Parent',ParentUI, 'Units','normalized', ...
                            'BackgroundColor', ParentColor, ...
                            'BorderType','none');
        obj.panel.set('ResizeFcn', ...
          @(src,evnt) SliderPanel.panelResize(obj,src,evnt));

        if(exist('Position','var') && ~isempty(Position))
          obj.panel.set('Position',Position);
        end
        
        if(exist('slideScale', 'var') && ~isempty(slideScale))
          obj.slideScale = slideScale;
        end

        %Text component (slider name)
        text_sep = .6;
        obj.text = uicontrol('Parent', obj.panel, ...
                             'Style','text','String',name, ...
                             'BackgroundColor',ParentColor);
        obj.text.HorizontalAlignment = 'center';
        obj.text.FontUnits = 'normalized';
        obj.text.FontSize = .5;
        obj.text.Units = 'normalized';
        obj.text.Position = [0 text_sep 1 1.-text_sep];


        %Create slider component
        [obj.slider,obj.sliderHG] = javacomponent( ...
          javax.swing.JSlider, [], obj.panel);

        obj.sliderHG.Units = 'normalized';
        obj.sliderHG.Position = [0 0 .85 text_sep];
        
        obj.slider.setPaintTicks(true);
        set(obj.slider, 'Minimum', min*obj.slideScale, ...
                        'Maximum', max*obj.slideScale);
        set(obj.slider, 'MinorTickSpacing',stepMinor*obj.slideScale, ...
                        'MajorTickSpacing',stepMajor*obj.slideScale, ...
                        'PaintLabels', true, ...
                        'Background', java.awt.Color(ParentColor(1), ...
                                                     ParentColor(2), ...
                                                     ParentColor(3)));

        %Create tick labels of the sllider
        labTable = java.util.Hashtable();
        for i = min:stepMajor:max
          lbl = javax.swing.JLabel(num2str(i));
          lbl.setFont(lbl.getFont().deriveFont(10));
          labTable.put(java.lang.Integer(int32(i*obj.slideScale)), lbl);
        end            
        obj.slider.setLabelTable(labTable);

        %Create value editable box
        obj.ValEditor = uicontrol('Parent', obj.panel, ...
                                  'Style', 'edit', ...
                                  'BackgroundColor', ParentColor);
        obj.ValEditor.Units = 'normalized';
        obj.ValEditor.Position = [.85, text_sep*.33, .15, text_sep*.33];

        %Initialize slider's value
        obj.setValue(val);

        %Set the callback function of the SliderPanel object
        obj.setCallbackFcn(callbackFcn);
        
      end


%%%%%%%%%%%%%%%%%%%%%%%%%     Access methods     %%%%%%%%%%%%%%%%%%%%%%%%%%
      function setValue(obj, value)
          if(value ~= obj.slider.value)
              % Set the slider's value without performing the actions in
              % callbackFromJSlider (these actions are only required when
              % the change of value is operated directly from slider's
              % action).
              obj.skipJSliderCallback = true;
              set(obj.slider, 'Value', ...
                  max(min(value*obj.slideScale, get(obj.slider,'Maximum')), ...
                      get(obj.slider,'Minimum')) );
          end

          set(obj.ValEditor, ...
              'String', num2str(round(value,obj.ValboxDecimalDigits)));
          obj.value = value;
      end


      function setCallbackFcn(obj, callbackFcn)
          if(isa(callbackFcn, 'function_handle'))
              obj.callbackFcn = callbackFcn;
              set(obj.slider, 'StateChangedCallback', @(src,evnt) SliderPanel.callbackFromJSlider(obj,src,evnt));
              set(obj.ValEditor, 'Callback', @(src,evnt) SliderPanel.callbackFromEditBox(obj,src,evnt));
          else
              obj.callbackFcn=[];
              set(obj.slider, 'StateChangedCallback', []);
              set(obj.ValEditor, 'Callback', []);
          end
      end


      % Implement abstract method from ListLayoutComponent class
      function setPosition(obj,pos)
        obj.panel.Position = pos;
      end
        

    end % methods
    
    
    
%%%%%%%%%%%%%%%%%%%%%%%     Callback functions     %%%%%%%%%%%%%%%%%%%%%%%%
    methods ( Static )
        
      %Resizing function
      function panelResize(obj,src,event)
      end

      function callbackFromJSlider(obj,src,event)
        if(~obj.skipJSliderCallback) % only true when the action comes
                                     % from the slider (to allow values
                                     % beyond the displayable slider
                                     % range without interfering with
                                     % other components).
          obj.value = obj.slider.value/obj.slideScale;
          obj.ValEditor.set('String',num2str(obj.value));
          if(~isempty(obj.callbackFcn))
            obj.callbackFcn(obj,event);
          end
        end
        obj.skipJSliderCallback = false;
      end

      function callbackFromEditBox(obj,src,event)
        val = str2double(obj.ValEditor.get('String'));
        if(isnan(val))
          obj.ValEditor.set('String',num2str(obj.value));
        else
          obj.setValue(val);
          if(~isempty(obj.callbackFcn))
            obj.callbackFcn(obj,event);
          end
        end
      end

    end % methods ( Static )

end