function plotTanh(trainedNetwork)
    
    % Parameters
    xMargin = 0.5;

    % Find quantization layer index
    for ii = 1:length(trainedNetwork.Layers)
        if isa(trainedNetwork.Layers(ii), 'QuantizationLayer')
            quantLayerInd = ii;
            break;
        end
    end
    
    % Extract coefs
    a = trainedNetwork.Layers(quantLayerInd).a';
    b = trainedNetwork.Layers(quantLayerInd).b';
    c = trainedNetwork.Layers(quantLayerInd).c';
    
    x = linspace(min(b)-xMargin, max(b)+xMargin, 1000);
    tanh_func = @(x) sum(meshgrid(a, x)' .* tanh(c.*(x - meshgrid(b, x)')), 1);
    
    q = zeros(1, length(x));
    for ii = 1:length(q)
        q(ii) = tanh2quantization(a, b, c, x(ii));
    end
                         
    plot(x, tanh_func(x), 'LineWidth', 2);
    hold on;
    plot(x, q, 'LineWidth', 2);
    grid on; grid minor; axis tight;
    zoom xon;
end