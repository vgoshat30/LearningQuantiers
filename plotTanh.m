function plotTanh(trainedNetwork)
    
    for ii = 1:length(trainedNetwork.Layers)
        if isequal(trainedNetwork.Layers(ii).Name, 'Quantization')
            quantLayerInd = ii;
            break;
        end
    end
    
    a = trainedNetwork.Layers(quantLayerInd).a';
    b = trainedNetwork.Layers(quantLayerInd).b';
    c = trainedNetwork.Layers(quantLayerInd).c';
    
    xMargin = 0.5;
    x = linspace(min(b)-xMargin, max(b)+xMargin, 10000);
    tanh_func = @(x) sum(meshgrid(a, x)' .* tanh(c .* (x - meshgrid(b, x)')), 1);
                         
    plot(x, tanh_func(x), 'LineWidth', 2);
    grid on; grid minor; axis tight;
    zoom xon;
end