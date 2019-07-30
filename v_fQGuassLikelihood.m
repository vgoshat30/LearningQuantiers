function v_fLikelihood = v_fQGuassLikelihood(m_fY, v_fMu, m_fCovW)

% Quantized Gaussian likelihood computation
%
% Syntax
% -------------------------------------------------------
% v_fLikelihood = v_fQGuassLikelihood(m_fY, v_fMu, m_fCovW)
%
% INPUT:
% -------------------------------------------------------
% m_fY - channel outputs
% v_fMu - mean vector
% m_fCovW - noise covariance
%
% OUTPUT:
% -------------------------------------------------------
% v_fLikelihood  - likelihood vector

% Quantization global parameters
global s_nLevels;
global s_fDynRange;

s_fEps =  1e-4;
s_fDelta = 2*s_fDynRange / s_nLevels;

s_fMaxVal = s_fDynRange - (s_fDelta/2);

% Order possible output values and corresponding regions
v_fQvals = -s_fMaxVal:s_fDelta:s_fMaxVal;
v_fRegLow = -s_fDynRange:s_fDelta:(s_fDynRange-s_fDelta);
v_fRegHigh = v_fRegLow + s_fDelta;
% Set edges to +- \inrty
v_fRegLow(1) = -100;
v_fRegHigh(end) = 100;

s_nSymbols = size(m_fY,2);

v_fLikelihood = zeros(1, s_nSymbols);


m_fYLowBound = zeros(size(m_fY));
m_fYUpBound = zeros(size(m_fY));
% Identify decision region valid for symbol
for ii=1:length(v_fQvals) 
    m_fYLowBound(find(abs(m_fY - v_fQvals(ii))<s_fEps)) = v_fRegLow(ii);
    m_fYUpBound(find(abs(m_fY - v_fQvals(ii))<s_fEps)) = v_fRegHigh(ii);
end  
 
% Compute liklihood of region
for kk=1:s_nSymbols
    v_fLikelihood(kk) =  mvncdf(m_fYLowBound(:,kk)', m_fYUpBound(:,kk)', v_fMu', m_fCovW);
end
