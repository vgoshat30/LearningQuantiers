function m_fShat = m_fGaussianML(m_fY, m_bDict, m_fH, m_fCovW)

% Maximum likelihood detector in Gaussian noise
%
% Syntax
% -------------------------------------------------------
% m_fShat = m_fGaussianML(m_fY, m_bDict, m_fH, m_fCovW)
%
% INPUT:
% -------------------------------------------------------
% m_fY - channel outputs
% m_bDict - input dictionary
% m_fH - channel matrix
% m_fCovW - noise covariance
%
% OUTPUT:
% -------------------------------------------------------
% m_fShat  - decoded values

global s_bQuantize;

s_nSymbols = size(m_fY, 2);
[s_nK, s_nDictSize] = size(m_bDict);

m_fShat = zeros(s_nK, s_nSymbols);
v_fLikelihood = zeros(1,s_nSymbols);
% Loop over dictionary
for ii=1:s_nDictSize
    % Compute likelihood for all symbols
    v_fMu = m_fH*m_bDict(:,ii);
    if (s_bQuantize == 0)   % Linear Gaussian channel
        v_fCurLikelihood = mvnpdf(m_fY', v_fMu', m_fCovW)';
    elseif (s_bQuantize == 1) % Quantized Gaussian channel
        v_fCurLikelihood = v_fQGuassLikelihood(m_fY, v_fMu, m_fCovW);
    end
    % Update symbols with improved likelihood over previous tested
    v_nUpdateIdx = find(v_fCurLikelihood > v_fLikelihood);
    m_fShat(:,v_nUpdateIdx) = repmat(m_bDict(:,ii),1,length(v_nUpdateIdx));
    % Save maximial likelihood values
    v_fLikelihood = max(v_fLikelihood, v_fCurLikelihood);
end


