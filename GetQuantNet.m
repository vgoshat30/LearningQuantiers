function Net = GetQuantNet(trainingSamples, traningLabels, quantizersNum, ...
                           codewordsNum, varargin)
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
    %
    %
    %
    %  Name-Value Pair Arguments:
    %
    %       'NetType'       -   Type of network:
    %                               'Class' for classification (default)
    %                               'Reg' for regression
    %
    %       'Epochs'        -   Maximum number of training epochs in each
    %                           repetition
    %
    %       'Repetitions'   -   Number of training iterations (the best one
    %                           is chosen)
    %
    %       'Plot'          -   Logical. Display a training progress plot or not
    
    %% Input Parser
    prsr = inputParser;
    prsr.CaseSensitive = false;
    
    defaultPlot = 0;
    defaultReps = 1; % Number of repetitions
    defaultEpochs = 1;
    defaultNetType = 'Class';
    
    addParameter(prsr, 'Plot', defaultPlot);
    addParameter(prsr, 'Repetitions', defaultReps); % Number of repetitions
    addParameter(prsr, 'Epochs', defaultEpochs);
    addParameter(prsr, 'NetType', defaultNetType); % 'Class' or 'Reg'
    
    parse(prsr, varargin{:});
    %% Settings
    if prsr.Results.Plot
        plotType = 'training-progress';
    else
        plotType = 'none';
    end
    
    options = trainingOptions('adam', ...
                'MaxEpochs', prsr.Results.Epochs, ...
                'Shuffle','every-epoch', ...
                'InitialLearnRate', 1e-3, ...
                'MiniBatchSize',64, ...
                'Verbose',false ...
                ,'Plots', plotType);
            
            
    s_nReps = prsr.Results.Repetitions;
    s_fLoss = inf;
    %% Create Layers
    inputDim = size(trainingSamples, 2);
    outputDim = size(traningLabels, 2);
    
    max_quant_in = max(trainingSamples, [], 'all');
    max_quant_out = max(traningLabels, [], 'all');
    
    trainSamplesCell = num2cell(trainingSamples', 1)';
    
    % Chose which layers will close the network according to NetType
    if isequal(prsr.Results.NetType, 'Class')
        traningLabels = categorical(traningLabels);
        outputDim = length(categories(traningLabels));
        closingLayers = [softmaxLayer; classificationLayer];
    elseif isequal(prsr.Results.NetType, 'Reg')
        outputDim = size(traningLabels, 2);
        closingLayers = regressionLayer;
    end
    
    layers = [ ...
        sequenceInputLayer(inputDim)
        lstmLayer(inputDim, 'OutputMode', 'last')
        fullyConnectedLayer(2*quantizersNum)
        reluLayer
        fullyConnectedLayer(quantizersNum)
        reluLayer
        QuantizationLayer(quantizersNum, codewordsNum, max_quant_in, max_quant_out)
        reluLayer
        fullyConnectedLayer(2*outputDim)
        reluLayer
        fullyConnectedLayer(outputDim)
        closingLayers
        ];
    %% Train network
    fprintf(['\nTraining network, ' num2str(s_nReps) ' iterations.\n']);
    fprintf('\tArchitecture:\n')
    fprintf(['\t\tQuantization Layer\t-->\tQuantizers: ' ...
             num2str(quantizersNum) '\t\tCodewords: ' ...
             num2str(codewordsNum) '\n']);
    fprintf('\tTraining Process:\n');
    
    
    % Train network multiple times and chose the one with best performance
    for kk=1:s_nReps
        fprintf(['\t\tTrain iteration ' num2str(kk) '...\n']);
        [tempNet, Info]  = trainNetwork(trainSamplesCell, traningLabels, layers, options);
        
        if mean(Info.TrainingLoss(end-floor(end/10):end)) < s_fLoss
            s_fLoss = mean(Info.TrainingLoss(end-floor(end/10):end));
            softNet = tempNet;
        elseif any(isnan(Info.TrainingLoss))
            warning('NaN Loss!');
        end
    end
    %% Convert to hard quantization
    % Find quantization layer index
    for ii = 1:length(softNet.Layers)
        if isa(softNet.Layers(ii), 'QuantizationLayer')
            quantLayerInd = ii;
            break;
        end
    end
    
    trainedLayers = softNet.Layers;
    
    % Sort quantization layers shifts and aply hard functions
    [b, I] = sort(trainedLayers(quantLayerInd).b);
    trainedLayers(quantLayerInd).a = trainedLayers(quantLayerInd).a(I);
    trainedLayers(quantLayerInd).b = b;
    trainedLayers(quantLayerInd).c = trainedLayers(quantLayerInd).c(I);
    trainedLayers(quantLayerInd) = HardQuantizationLayer(trainedLayers(quantLayerInd));
    
    % Return network
    Net = assembleNetwork(trainedLayers);
end