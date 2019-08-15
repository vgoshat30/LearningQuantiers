function VisualizeNet(Net, ShowMenu) %#ok<INUSD>
    % VISUALIZENET gets a trained network as returned from GetQuantNet or 
    % GetADCNet and plots the soft and hard quantization (or sampling) functions
    % 
    %
    % Input:
    %
    %       Net - The trained network as returned from GetQuantNet or GetADCNet
    
    %% Exctract coefficients
    % Find quantization layer index
    quantLayerInd = NaN;
    samplLayerInd = NaN;
    for ii = 1:length(Net.Layers)
        if isa(Net.Layers(ii), 'HardQuantizationLayer')
            quantLayerInd = ii;
        elseif isa(Net.Layers(ii), 'HardSamplingLayer')
            samplLayerInd = ii;
        end
    end
    
    if isnan(quantLayerInd)
        error('No quantization layer exists.');
    end
    %% Create sampling layer functions
    if ~isnan(samplLayerInd)
        phi = Net.Layers(samplLayerInd).phi';
        sigma = Net.Layers(samplLayerInd).sigma';
        T = Net.Layers(samplLayerInd).T;

        xSampl = linspace(min([0 phi(1)-2*sigma(1)]), ...
                          max([T phi(end)+2*sigma(end)]), 1000);
    
        gaussFunc = @(x) sum(exp(-((meshgrid(xSampl, phi)-meshgrid(phi, x)') ...
                             ./meshgrid(sigma, x)').^2), 1);
    end
    %% Create quantization layer functions
    % Extract coefs
    a = Net.Layers(quantLayerInd).a';
    b = Net.Layers(quantLayerInd).b';
    c = Net.Layers(quantLayerInd).c';
    
    % Create quantization functions
    bdiff = max(diff(b));
    xQuant = linspace(min(b)-bdiff, max(b)+bdiff, 1000);
    tanh_func = @(x) sum(meshgrid(a, x)' .* tanh(c.*(x - meshgrid(b, x)')), 1);
    
    q = zeros(1, length(xQuant));
    for ii = 1:length(q)
        q(ii) = tanh2quantization(a, b, c, xQuant(ii));
    end
    %% Plot
    if nargin == 2
        figMenu = 'Figure';
    else
        figMenu = 'None';
    end
    
    
    if ~isnan(samplLayerInd)
        figure('Name', 'Soft and Hard Sampling and Quantization Functions', ...
           'NumberTitle', 'off', 'Menu', figMenu, 'Units', 'Normalized', ...
           'Position', [0.1 0.2 0.8 0.5]);
        subplot(5, 2, [3 5 7]);
        plot(xSampl, gaussFunc(xQuant), 'LineWidth', 2, 'DisplayName', 'Soft');
        hold on;
        stem(phi, ones(size(phi)), 'LineWidth', 2, 'DisplayName', 'Samples');
        grid on; grid minor; axis tight; zoom on;
        ylim([0 1.5]);
        legend('Show', 'Interpreter', 'LaTex', 'Location', 'Best');
        xlabel('Dense Samples','Interpreter','LaTex','Fontsize',15);
        title('Sampling Layer','Interpreter','LaTex','Fontsize',20);
        subplot(5, 2, [2 4 6 8 10]);
    else
        figure('Name', 'Soft and Hard Quantization Functions', ...
           'NumberTitle', 'off', 'Menu', figMenu);
    end
    plot(xQuant, tanh_func(xQuant), 'LineWidth', 2, 'DisplayName', 'Soft');
    hold on;
    plot(xQuant, q, 'LineWidth', 2, 'DisplayName', 'Hard');
    grid on; grid minor; axis tight; zoom on;
    legend('Show', 'Interpreter', 'LaTex', 'Location', 'Best');
    xlabel('Input of Quantizer','Interpreter','LaTex','Fontsize',15);
    ylabel('Output of Quantizer','Interpreter','LaTex','Fontsize',15);
    title('Quantization Layer','Interpreter','LaTex','Fontsize',20);
end