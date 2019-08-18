% Test channel estimation data from Imperial College
%% Prepare Workspace
clear ChannelEstimTest1 variables; close all; clc;

% Load Data
load('ChannelEstimationData/DLHFIN_Ch2.mat');
% load('data_PIC.mat');

% Normalize data
maxHFDL = max(HFDL, [], 'all');
HFDL = HFDL / maxHFDL;
setSize = size(HFDL, 1);
%% Generate Channel Outputs Data
%
% Assumed here that the third dimension of HFDL is the base station antennas and
% the fourth one is the users -> should re-check
nt = 32; % max is 256
nu = 4; % max is 32

meas2paramRatio = 4;

power = 1;

% Pilots matrixes
tau = nu * meas2paramRatio;  
Phi = dftmtx(tau);
Phi = Phi(:,1:nu);

% Channel Vectorization
H = HFDL(:, 1, 1:nt, 1:nu) + 1j * HFDL(:, 1, 1:nt, 1:nu);
H = permute(H, [1 3 4 2]);
H = H(:, :).';

% Gaussian Noise
W = 1/sqrt(2) * (randn(tau * nt, setSize) + 1j*randn(tau * nt,  setSize));

% Channel outputs
Y = sqrt(power) *(kron(Phi, eye(nt))) * H + W;

labels = [real(H); imag(H)].';
data = [real(Y); imag(Y)].';
%
%% Create, Train and Test Network

trainingPortion = 0.5;

% Divide data
trainSamplesNum = floor(setSize * trainingPortion);
% Train
trainData = data(1:trainSamplesNum, :);
trainLabels = labels(1:trainSamplesNum, :);
% Test
testData = data(1:(setSize-trainSamplesNum), :);
testLabels = labels(1:(setSize-trainSamplesNum), :);
%}

% trainData = trainX;
% trainLabels = trainS;
% testData = dataX;
% testLabels = dataS;

mse = [];
quantNet = [];
rate = [];
testNum = 5;
quantizersList = 5:6;
codewordsList = 3:4;

fig = figure;
dcm_obj = datacursormode(fig);
datacursormode on
set(dcm_obj,'UpdateFcn',@textUpdateFun);

ax = axes; grid on; grid minor; hold on;
clr = get(ax, 'ColorOrder');
clr = clr(1, :);
xlabel('Rate', 'Interpreter', 'LaTex', 'FontSize', 20);
ylabel('Loss', 'Interpreter', 'LaTex', 'FontSize', 20);



for quantizersInd = 1:length(quantizersList)
    for codewordsInd = 1:length(codewordsList)
        quantizers = quantizersList(quantizersInd);
        codewords = codewordsList(codewordsInd);
        rate(end+1) = quantizers * codewords / size(labels, 2); %#ok<SAGROW>
        %% Train
        quantNet = [quantNet GetQuantNet(trainData, trainLabels, quantizers, ...
                    codewords, 'NetType', 'Reg')]; %#ok<AGROW>
        %% Test
        fprintf('Testing Network.\t');
        SPredicted = predict(quantNet(end), num2cell(testData', 1));
        mse(end+1) = mean(mean((SPredicted - testLabels).^2, 2), 1); %#ok<SAGROW>
        fprintf(['MSE:\t' num2str(mse(end)) '\n']);
        plot(rate(end), mse(end), 'x', 'Color', clr, 'LineWidth', 2, ...
             'MarkerSize', 10, ...
             'DisplayName', num2str(quantizers) + " " + num2str(codewords));
        drawnow;
    end
end
%% Save Results
clear HFDL ax dcm_obj fig clr codewords quantizers quantizersInd ...
      codewordsInd setSize;
save("Results/results " + datestr(datetime) + ".mat");
savefig("Results/results " + datestr(datetime) + ".fig");