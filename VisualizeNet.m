function VisualizeNet(Net)
    % VISUALIZENET gets a trained network as returned from GetQuantNet and
    % plots the soft and hard quantization functions
    % 
    %
    % Input:
    %
    %       Net -   The trained network as returned from GetQuantNet
    
    %% Exctract coefficients
    % Find quantization layer index
    for ii = 1:length(Net.Layers)
        if isa(Net.Layers(ii), 'HardQuantizationLayer')
            quantLayerInd = ii;
            break;
        end
    end
    
    % Extract coefs
    a = Net.Layers(quantLayerInd).a';
    b = Net.Layers(quantLayerInd).b';
    c = Net.Layers(quantLayerInd).c';
    %% Create quantization functions
    bdiff = max(diff(b));
    x = linspace(min(b)-bdiff, max(b)+bdiff, 1000);
    tanh_func = @(x) sum(meshgrid(a, x)' .* tanh(c.*(x - meshgrid(b, x)')), 1);
    
    q = zeros(1, length(x));
    for ii = 1:length(q)
        q(ii) = tanh2quantization(a, b, c, x(ii));
    end
    %% Plot
    figure('Name', 'Soft and Hard Quantization Functions', ...
           'NumberTitle', 'off', 'Menu', 'None');
    plot(x, tanh_func(x), 'LineWidth', 2, 'DisplayName', 'Soft');
    hold on;
    plot(x, q, 'LineWidth', 2, 'DisplayName', 'Hard');
    grid on; grid minor; axis tight; zoom on;
    legend('Show', 'Interpreter', 'LaTex', 'Location', 'Best');
    xlabel('Input of Quantization Layer','Interpreter','LaTex','Fontsize',15);
    ylabel('Output of Quantization Layer','Interpreter','LaTex','Fontsize',15);
end