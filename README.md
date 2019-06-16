# LearningQuantiers

Functions to use:
* Net = GetQuantNet(trainingSamples, traningLabels, quantizersNum, codewordsNum)
* VisualizeNet(Net)

Example:
```
load('data_PIC.mat');
Net = GetQuantNet(trainX, trainS, 5, 10);
VisualizeNet(Net)
```
