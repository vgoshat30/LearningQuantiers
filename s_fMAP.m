function s_fErr = s_fMAP(m_fY, m_fS, m_fH, m_fCovW)
% Maximum aposteriori probability detector (equiprobable BPSK constellation)
%
% Syntax
% -------------------------------------------------------
% s_fErr = s_fMAP(m_fY, m_fS, m_fH, m_fCovW)
%
% INPUT:
% -------------------------------------------------------
% m_fY - channel outputs
% m_fS - channel inputs
% m_fH - channel matrix
% m_fCovW - noise covariance
%
% OUTPUT:
% -------------------------------------------------------
% s_fErr  - average bit error rate

s_nK = size(m_fH,2);

% Generate possible input vectors
m_bInputs = -1*ones(s_nK, 2^s_nK);
for val=2:2^s_nK
    kk = val-1;
    idx = 1;
    while (kk > 0) 
        if(mod(kk,2)) m_bInputs(idx,val) = 1; end
        kk = floor(kk/2);
        idx = idx+1;
    end
end

% Compute ML for each input
m_fShat = m_fGaussianML(m_fY, m_bInputs, m_fH, m_fCovW);

% Compute BER
s_fErr = mean(mean(m_fShat ~= m_fS));
