function output_txt = textUpdateFun(~,event_obj)
% Display the position of the data cursor
% obj          Currently not used (empty)
% event_obj    Handle to event object
% output_txt   Data cursor text string (string or cell array of strings).

pos = get(event_obj,'Position');

presicion = 5;

dispName = event_obj.Target.DisplayName;
dispName = split(dispName);

output_txt = {['Rate: ', num2str(pos(1), presicion)], ...
              ['Loss: ', num2str(pos(2), presicion)], ...
              ['Quantizers: ', dispName{1}], ...
              ['Codewords: ', dispName{2}]};