function Net = GetQuantNet(trainingSamples, traningLabels, quantizersNum, codewordsNum)
    % GETQUANTNET trains a quantization network over given data with soft
    % quantization function and terurns a trained network with hard
    % quantization function
    % 
    %
    % Inputs:
    %
    %       trainingSamples -   NxM double array, where N is the samples
    %                           amount and M is the input dimension
    %
    %       traningLabels   -   NxK double array, where N is the samples
    %                           amount and K is the output dimension
    %
    %       quantizersNum   -   Number of identical quantizers at the
    %                           quantization layer
    %
    %       codewordsNum    -   Number of codewords in each quantizer
    %
    %
    % Output:
    %
    %       Net             -   Trained network with hard quantization function
    
    %% Settings
    options = trainingOptions('sgdm', ...
                'MaxEpochs',100, ...
                'Shuffle','every-epoch', ...
                'InitialLearnRate', 1e-2, ...
                'MiniBatchSize',64, ...
                'Plots', 'training-progress');
    
    % layersConfig is a cell array specifying the architecture of the network.
    % Supported layers are fullyConnected relu and quantization.
    % The syntax of each is:
    %
    %   'fullyConnected x'  -   Where x is a multiplier specifying the
    %                           multiplication factor between the input and the
    %                           output of the layer.
    %                           For example: 'fullyConnected 1.5'
    %
    %   'relu'              -   ReLU layer
    % 
    %   'Quantization'      -   Quantization layer with the parameters
    %                           specified in GetQuantNet
    %
    % If you want to define the layers manually yourself,
    % use 'Manually create layers' section below
    layersConfig = {
                    'fullyConnected 1.5', ...
                    'relu', ...
                    'fullyConnected 1.5', ...
                    'relu', ...
                    'fullyConnected 0.75', ...
                    'relu', ...
                    'fullyConnected 0.75', ...
                    'relu', ...
                    'Quantization', ...
                    'relu', ...
                    'fullyConnected 2', ...
                    'relu', ...
                    'fullyConnected 0.5', ...
                    'relu', ...
                    'fullyConnected 0.5', ...
                    'relu', ...
                    'fullyConnected 0.5', ...
                    };
    %% Check formatting of input data
    if ~isa(trainingSamples, 'double')
        error("trainingSamples is of type " + class(trainingSamples) + ...
              " but sould be double.");
    end
    if ~isa(traningLabels, 'double')
        error("trainingSamples is of type " + class(trainingSamples) + ...
              " but sould be double.");
    end
    if size(trainingSamples,1) ~= size(traningLabels,1)
        error("First dimension of trainingSamples and traningLabels" + ...
              " sould be equal. This is the samples amount.");
    end
    %% Create Layers
    inputDim = size(trainingSamples, 2);
    outputDim = size(traningLabels, 2);
    
    % Find last fully connected layer
    lastFullyConnInd = [];
    for ii = length(layersConfig):-1:1
        if ~isempty(strfind(layersConfig{ii}, 'fullyConnected'))
            lastFullyConnInd = ii;
            break;
        end
    end
    if isempty(lastFullyConnInd)
        error("Specify at least one fullyConnected layer in layersConfig" + ...
              " in Settings.");
    end
    
    % Add first two layers
    layers = [ ...
        sequenceInputLayer(inputDim)
        lstmLayer(inputDim, 'OutputMode', 'last')];
    
    % Iterate over layersConfig and add all layers
    lastOutDim = inputDim;
    hasQuantizationLayer = false;
    for ii = 1:length(layersConfig)
        if ~isempty(strfind(layersConfig{ii}, 'fullyConnected'))
            if ii+1 < length(layersConfig) && ...
               ~isempty(strfind(layersConfig{ii+1}, 'Quantization'))
                lastOutDim = quantizersNum;
            elseif ii == lastFullyConnInd
                lastOutDim = outputDim;
            else
                splitStr = strsplit(layersConfig{ii}, ' ');
                if length(splitStr) == 1
                    error("Check syntax of layersConfig{" + num2str(ii) + ...
                          "}" + " in Settings.");
                end
                lastOutDim = floor(lastOutDim * str2double(splitStr{2}));
            end
            layers = [layers; fullyConnectedLayer(lastOutDim)]; %#ok<AGROW>
        elseif ~isempty(strfind(layersConfig{ii}, 'relu'))
            layers = [layers; reluLayer]; %#ok<AGROW>
        elseif ~isempty(strfind(layersConfig{ii}, 'Quantization'))
            layers = [layers; QuantizationLayer(quantizersNum, codewordsNum)]; %#ok<AGROW>
            hasQuantizationLayer = true;
        end
    end
    if ~hasQuantizationLayer
        error("Quantization layer not specified in layersConfig in Settings.");
    end
    % Add last layer
    layers = [layers; regressionLayer];
    %% Manually create layers
%     layers = [ ...
%         sequenceInputLayer(inputDim)
%         lstmLayer(inputDim, 'OutputMode', 'last')
%         fullyConnectedLayer(quantizersNum)
%         reluLayer
%         QuantizationLayer(quantizersNum, codewordsNum)
%         reluLayer
%         fullyConnectedLayer(outputDim)
%         reluLayer
%         fullyConnectedLayer(outputDim)
%         regressionLayer
%         ];
    %% Train network
    trainSamplesCell = num2cell(trainingSamples', 1)';
    softNet = trainNetwork(trainSamplesCell, traningLabels, layers, options);
    %% Convert to hard quantization
    % Find quantization layer index
    for ii = 1:length(softNet.Layers)
        if isa(softNet.Layers(ii), 'QuantizationLayer')
            quantLayerInd = ii;
            break;
        end
    end
    
    trainedLayers = softNet.Layers;
    trainedLayers(quantLayerInd) = HardQuantizationLayer(trainedLayers(quantLayerInd));

    Net = assembleNetwork(trainedLayers);
end