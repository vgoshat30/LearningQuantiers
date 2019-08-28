# LearningQuantiers

Functions to use:
* Net = GetQuantNet(trainingSamples, traningLabels, quantizersNum, codewordsNum, varargin)
* Net = GetADCNet(trainingSamples, traningLabels, quantizersNum, codewordsNum, observedT, samplesNum, varargin)
* VisualizeNet(Net)

Example:
```
Net = GetADCNet(m_fYtrain', v_fDtrain', s_nP, codewordsNum, ...
                s_nT, s_nTtilde, 'NetType', 'Class', ...
                'Repetitions', 2, 'Epochs', 5, 'Plot', 0);
VisualizeNet(Net)
```
