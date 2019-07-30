function Net = GetFixQuantNet(trainingSamples, traningLabels, quantizersNum, codewordsNum)
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
    options = trainingOptions('adam', ...
                'MaxEpochs',100, ...
                'Shuffle','every-epoch', ...
                'InitialLearnRate', 1e-2, ...
                'MiniBatchSize',64, ...
                'Verbose',false ...
                ); %,'Plots', 'training-progress');
            
            
    s_nReps = 1;        % Number of repetitions
    s_fLoss = inf;
   
    %% Create Layers
    inputDim = size(trainingSamples, 2);

    
    trainSamplesCell = num2cell(trainingSamples', 1)';
    % Added for classification networks
    traningLabelsCat = categorical(traningLabels);
    outputDim = length(categories(traningLabelsCat));
    %% Manually create layers
    layers = [ ...
        sequenceInputLayer(inputDim)
        lstmLayer(inputDim, 'OutputMode', 'last')
        fullyConnectedLayer(quantizersNum)
%        reluLayer
        FixedQuantizationLayer(quantizersNum, codewordsNum) 
        fullyConnectedLayer(outputDim)
        reluLayer
        fullyConnectedLayer(outputDim)
        softmaxLayer
        classificationLayer
        ];
    %% Train network
    for kk=1:s_nReps        
        [tempNet, Info]  = trainNetwork(trainSamplesCell, traningLabelsCat, layers, options);
        if (mean(Info.TrainingLoss(end-1000:end)) < s_fLoss)
            s_fLoss = mean(Info.TrainingLoss(end-1000:end));
            softNet = tempNet;
        end
    
    end
    %% Convert to hard quantization
    % Find quantization layer index
    for ii = 1:length(softNet.Layers)
        if isa(softNet.Layers(ii), 'FixedQuantizationLayer')
            quantLayerInd = ii;
            break;
        end
    end
          
    
    trainedLayers = softNet.Layers; 
     % sort by shifts - not done in training
    [b, I] = sort(trainedLayers(quantLayerInd).b);
    trainedLayers(quantLayerInd).a = trainedLayers(quantLayerInd).a(I);
    trainedLayers(quantLayerInd).b = b;
    trainedLayers(quantLayerInd).c = trainedLayers(quantLayerInd).c(I);
    trainedLayers(quantLayerInd) = HardQuantizationLayer(trainedLayers(quantLayerInd));

    Net = assembleNetwork(trainedLayers);
end