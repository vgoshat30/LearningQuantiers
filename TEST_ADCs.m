% Test deep task-based quantization for estimation
clear variables;
close all;
clc;

rng(2);
global s_nLevels;
global s_fDynRange;
global s_bQuantize;
%% Parameters setting
s_nN  = 12;             % Number of Rx antennas
s_nT = 10;              % Observed time frame
s_nK  = 4;             % Number of transmitted symbols
s_fTrainSize = 5000;    % Training size
s_fTestSize = 5000;    % Test data size
s_nP = s_nK;            % Number of scalar quantizers
s_nTtilde = 2;          % Number of samples to produce over time frame
s_nChannels = 1;         % 1 - Gaussian channel; 

s_fEstErrVar = 0.2;   % Estimation error variance
% Frame size for generating noisy training
s_fFrameSize = 500; 
s_fNumFrames = s_fTrainSize/s_fFrameSize;
s_fNumTestFrames = s_fTestSize/s_fFrameSize;

v_fSNRdB =  6:14; % [8 12];    %  % SNR values in dB.
v_fRate =  1; %[1,2]; %1:0.2:3;%[1,2];   % Quantization rate

s_nXaxis = 1;       % 1 - SNR
                    % 2 - Quantization rate

switch s_nXaxis
    case 1
        v_fX = v_fSNRdB;
        v_fY = v_fRate;   
        stXaxis = 'SNR [dB]';
    case 2
        v_fY = v_fSNRdB;
        v_fX = v_fRate;   
        stXaxis = 'Quantization rate';
end

% Select which decoder to simulate
v_nCurves   = [...          % Curves
    0 ...                   % MAP, perfect CSI
    0 ...                   % MAP, CSI uncertainty
    1 ...                   % DNN, Soft-to-hard, optimal training  
    0 ...                   % DNN, Soft-to-hard, CSI uncertainty
    0 ...                   % DNN, Passing gradient, optimal training  
    0 ...                   % DNN, Passing gradient, CSI uncertainty
    0 ...                   % Quantized MAP, perfect CSI  
    0 ...                   % DNN, Soft-to-hard fixed, optimal training  
    ];

s_nCurves = length(v_nCurves);

v_stPlots = strvcat(  ...
    'MAP, perfect CSI', ...
    'MAP, CSI uncertainty',...
    'DNN, Soft-to-hard, optimal training',... 
    'DNN, Soft-to-hard, CSI uncertainty', ...
    'DNN, Passing gradient, optimal training',... 
    'DNN, Passing gradient, CSI uncertainty', ...
    'Quantized MAP, perfect CSI', ...
    'DNN, uniform soft-to-hard, optimal training'... 
    );

% Generate channel matrix
m_fH = zeros(s_nN, s_nK);
for ii=1:s_nN
    for jj=1:s_nK
        m_fH(ii,jj) = exp(-abs(ii-jj));
    end
end
% Generate time diversity vectors
v_fChannelTD = 1+0.5*cos(1:s_nT);
v_fNoiseTD = 1+0.3*cos(1:s_nT + 0.2);

if(s_nChannels == 1)
    % BPSK constellation
    v_fConst = [-1 1];
    fSymToProb = @(x)0.5*(x+1);  
end
%% Simulation loop
m_fBER = zeros(s_nCurves, length(v_fX), length(v_fY));

% Generate training symbols - BPSK
m_fStrain = randsrc(s_nK,s_fTrainSize, v_fConst);

% Generate test symbols - BPSK
m_fStest = randsrc(s_nK,s_fTestSize, v_fConst);

% Convert to decision vectors
v_fDtrain = zeros(1,s_fTrainSize);
v_fDtest = zeros(1,s_fTestSize);
for ii=1:s_nK
    v_fDtrain = v_fDtrain + 0.5*(1+m_fStrain(ii,:)) * 2^(ii-1);
    v_fDtest = v_fDtest + 0.5*(1+m_fStest(ii,:)) * 2^(ii-1);
end
v_fDtest = categorical(v_fDtest);

% Training with noisy CSI
m_fRtrain = zeros(s_nN, s_fTrainSize);
for kk=1:s_fNumFrames
    Idxs=((kk-1)*s_fFrameSize + 1):kk*s_fFrameSize;
    m_fRtrain(:,Idxs) =  (m_fH.*(1+  sqrt(s_fEstErrVar)*randn(size(m_fH))))*m_fStrain(:,Idxs);
end
%%

for ii=1:length(v_fSNRdB)
    s_bQuantize = 0;
    s_fSigW = 10^(-0.1* v_fSNRdB(ii));
    s_fMAPBER1 = 0;
    s_fMAPBER2 = 0;
    m_fYtrain = [];
    m_fYtrainErr = [];
    m_fYtest = [];
    % Generate channel outputs
    if((s_nChannels == 1))
        for tt=1:s_nT
        m_fYtrain =[m_fYtrain; ...
                    v_fChannelTD(tt)*m_fH * m_fStrain + sqrt(s_fSigW*v_fNoiseTD(tt))*randn(s_nN, s_fTrainSize)];
        m_fYtrainErr = m_fRtrain + sqrt(s_fSigW)*randn(s_nN, s_fTrainSize);
        m_fYtest = [m_fYtest; ...
                        v_fChannelTD(tt)*m_fH * m_fStest + sqrt(s_fSigW*v_fNoiseTD(tt))*randn(s_nN, s_fTestSize)];
        end
    end
    
    % MAP detector
    if (v_nCurves(1) == 1)
        s_fMAPBER1 = s_fMAP(m_fYtest, m_fStest, m_fH, (s_fSigW)*eye(s_nN));
    end
    
    % MAP detector, uncertainty
    if (v_nCurves(2) == 1)
        for jj=1:s_fNumTestFrames
            v_fIdxs = ((jj-1)*s_fFrameSize+1) : (jj*s_fFrameSize);
            s_fMAPBER2 = s_fMAPBER2 + s_fMAP(m_fYtest(:,v_fIdxs), m_fStest(:,v_fIdxs), ...
                                                (m_fH.*(1+  sqrt(s_fEstErrVar)*randn(size(m_fH)))),...
                                                (s_fSigW)*eye(s_nN));
        end
        % Average over noisy channel tests
        s_fMAPBER2 = s_fMAPBER2/s_fNumTestFrames;
    end
    
    for jj=1:length(v_fRate)
        % TODO NIR REMOVE DEBUG
        s_nP = floor((s_nN/3) * v_fRate(jj));
        codewordsNum = floor(2^(v_fRate(jj) * s_nN /s_nP));
        switch s_nXaxis
            case 1
                s_fXidx = ii;
                s_fYidx = jj;                
            case 2
                s_fXidx = jj;
                s_fYidx = ii; 
        end
        m_fBER(1,s_fXidx,s_fYidx) = s_fMAPBER1;
        m_fBER(2,s_fXidx,s_fYidx) = s_fMAPBER2;
        
        % Deep task-based quantizer
        if (v_nCurves(3) == 1)                  
            % Get network
            % TO GOSHA - you will need to add as an input to this function
            % also the parameter snTtilde which determines how many samples
            % to take per time interval, and perhaps also the time interval
            % duration s_nT
            v_cNet = GetADCNet(m_fYtrain', v_fDtrain', s_nP, codewordsNum, ...
                                 s_nT, s_nTtilde, 'NetType', 'Class', ...
                                 'Repetitions', 1, 'Epochs', 1, 'Plot', 0);     
            
            % Apply network
            v_fDhat = classify(v_cNet,num2cell(m_fYtest, 1)')';
            % Convert classification into symbols           
            m_fBhat = de2bi(double(string(v_fDhat)))';
            m_fShat = -1*ones(size(m_fStest)) + 2*m_fBhat;
            % Compute error
            m_fBER(3,s_fXidx,s_fYidx)  =  mean(mean(m_fShat ~= m_fStest));
            fprintf(['\tTest results:\n\t\tError = ' num2str(m_fBER(3,s_fXidx,s_fYidx)) '\n']);
        end


        % Deep task-based quantizer, uncertainty
        if (v_nCurves(4) == 1)        
            % Get network
            v_cNet = GetQuantNet(m_fYtrainErr', v_fDtrain', s_nP, codewordsNum);
            
            % Apply network
            v_fDhat = classify(v_cNet,num2cell(m_fYtest, 1)')';
            % Convert classification into symbols           
            m_fBhat = de2bi(double(string(v_fDhat)))';
            m_fShat = -1*ones(size(m_fStest)) + 2*m_fBhat;
            % Compute error
            m_fBER(4,s_fXidx,s_fYidx)  = mean(mean(m_fShat ~= m_fStest));

        end
        
       % Deep task-based quantizer passing gradient
        if (v_nCurves(5) == 1)                  
            % Get network
            v_cNet = GetQuantNoiseNet(m_fYtrain', v_fDtrain', s_nP, codewordsNum);     
            
            % Apply network
            v_fDhat = classify(v_cNet,num2cell(m_fYtest, 1)')';
            % Convert classification into symbols           
            m_fBhat = de2bi(double(string(v_fDhat)))';
            m_fShat = -1*ones(size(m_fStest)) + 2*m_fBhat;
            % Compute error
            m_fBER(5,s_fXidx,s_fYidx)  =  mean(mean(m_fShat ~= m_fStest));

        end


        % Deep task-based quantizer, passing gradient, uncertainty
        if (v_nCurves(6) == 1)        
            % Get network
            v_cNet = GetQuantNoiseNet(m_fYtrainErr', v_fDtrain', s_nP, codewordsNum);
            
            % Apply network
            v_fDhat = classify(v_cNet,num2cell(m_fYtest, 1)')';
            % Convert classification into symbols           
            m_fBhat = de2bi(double(string(v_fDhat)))';
            m_fShat = -1*ones(size(m_fStest)) + 2*m_fBhat;
            % Compute error
            m_fBER(6,s_fXidx,s_fYidx)  = mean(mean(m_fShat ~= m_fStest));

        end        
        
        % Quantized MAP detector
        if (v_nCurves(7) == 1)
            s_bQuantize = 1;
            % Quantization parameters
            s_fDynRange = 2;
            s_nLevels = floor(2^(v_fRate(jj)));
            m_fBER(7,s_fXidx,s_fYidx) = s_fMAP(m_fQuant(m_fYtest, s_nLevels, s_fDynRange),...
                                               m_fStest, m_fH, (s_fSigW)*eye(s_nN));
        end
        
        % Deep fixed task-based quantizer
        if (v_nCurves(8) == 1)                  
            % Get network
            v_cNet = GetFixQuantNet(m_fYtrain', v_fDtrain', s_nP, codewordsNum);     
            
            % Apply network
            v_fDhat = classify(v_cNet,num2cell(m_fYtest, 1)')';
            % Convert classification into symbols           
            m_fBhat = de2bi(double(string(v_fDhat)))';
            m_fShat = -1*ones(size(m_fStest)) + 2*m_fBhat;
            % Compute error
            m_fBER(8,s_fXidx,s_fYidx)  =  mean(mean(m_fShat ~= m_fStest));

        end        
 
    end
end
%% Display results
 v_stPlotType = strvcat( '-rs', '--ro', '-b^',  '--bv', '-k<', '--k>',...
     '-m*', '-mx',  '-c^', '--cv');


%

for kk=1:length(v_fY)
    v_stLegend = [];
    fig1 = figure;
    set(fig1, 'WindowStyle', 'docked');
%     switch s_nXaxis
%         case 1
%             stCurLegend = [', R = ', num2str(v_fRate(kk))];
%         case 2
%             stCurLegend = [', SNR = ', num2str(v_fSNRdB(kk))];
%     end
    stCurLegend =[];
    for aa=1:s_nCurves
        if (v_nCurves(aa) ~= 0)
            s_nIdx =  aa;
            v_stLegend = strvcat(v_stLegend,  [v_stPlots(aa,:), stCurLegend]);
            semilogy(v_fX, m_fBER(aa,:,kk), v_stPlotType(s_nIdx,:),'LineWidth',1,'MarkerSize',10);
            hold on;
        end
    end
    xlabel(stXaxis);
    ylabel('BER');
    grid on;
    legend(v_stLegend,'Location','SouthWest');
 %   set(gca,'FontSize',12);
    hold off;
end
%% Finish Alarm
load handel;
sound(y,Fs);


