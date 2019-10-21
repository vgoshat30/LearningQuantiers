function delete_all_fig()
    % Call to close all figures created during call to TEST_ChannelEstimQuant
    % if it was terminated using ctrl + C
    delete(findobj('type','figure'));
end