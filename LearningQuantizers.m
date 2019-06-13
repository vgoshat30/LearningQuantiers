%% Prepare Workspace
clear LearningQuantizers QuantizationLayer; close all; clc;
%% Parameters
DATA_FILENAME = 'data_PIC.mat';

VALIDATION_PERCENT = 0.1;
%% Prepare Data
load(DATA_FILENAME);
%% Define Network
layers = [ ...
    sequenceInputLayer(12)
    lstmLayer(12, 'OutputMode', 'last')
    fullyConnectedLayer(20)
    reluLayer
    fullyConnectedLayer(10)
    reluLayer
    QuantizationLayer(10, 5)
    reluLayer
    fullyConnectedLayer(6)
    regressionLayer
    ];


options = trainingOptions('sgdm', ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'InitialLearnRate', 1e-3, ...
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
% TODO: ***** Figure out a way to test the network with hard quantization
% TODO: ****_ Save few tests and see if incrementing codewords num lowers loss
SPredicted = predict(trainedNet, num2cell(dataX', 1));
mse = mean(mean((SPredicted - dataS).^2, 2), 1);
plotTanh(trainedNet);





