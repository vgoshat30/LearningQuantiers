% This script was used before GetQuantNet and is kept for any case...
%% Prepare Workspace
clear LearningQuantizers QuantizationLayer; close all; clc;

% Prepare Data
load('data_PIC.mat');
%% Define Network
layers = [ ...
    sequenceInputLayer(12)
    lstmLayer(12, 'OutputMode', 'last')
    fullyConnectedLayer(15)
    reluLayer
    fullyConnectedLayer(20)
    reluLayer
    fullyConnectedLayer(25)
    reluLayer
    fullyConnectedLayer(15)
    reluLayer
    fullyConnectedLayer(10)
    reluLayer
    QuantizationLayer(10, 6)
    reluLayer
    fullyConnectedLayer(15)
    reluLayer
    fullyConnectedLayer(6)
    regressionLayer
    ];


options = trainingOptions('sgdm', ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'InitialLearnRate', 1e-2, ...
...%     'LearnRateSchedule', 'piecewise', ...
...%     'LearnRateDropPeriod', 100, ...
...%     'LearnRateDropFactor', 0.95, ...
    'MiniBatchSize',64, ...
    'Plots','training-progress');
%% Train Network
trainIn = num2cell(trainX', 1)';
trainOut = num2cell(trainS', 1)';
[trainedNet, traininfo] = trainNetwork(trainIn, trainS, layers, options);
%% Test Network

% Find quantization layer index
for ii = 1:length(trainedNet.Layers)
    if isa(trainedNet.Layers(ii), 'QuantizationLayer')
        quantLayerInd = ii;
        break;
    end
end

trainedLayers = trainedNet.Layers;
trainedLayers(quantLayerInd) = HardQuantizationLayer(trainedLayers(quantLayerInd));

hardQuantNet = assembleNetwork(trainedLayers);

SPredicted = predict(hardQuantNet, num2cell(dataX', 1));
mse = mean(mean((SPredicted - dataS).^2, 2), 1);
VisualizeNet(hardQuantNet);






