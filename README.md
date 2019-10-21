# LearningQuantiers

## Get[...]Net Functions

Each function with a Get prefix creates and trains a network, according to its name.
All the functions with the Get prefix control whether to display a learning plot or not, via the Name-Value parameter 'Plot'.

## TEST_[...] Scripts

Scripts to run are with the prefix TEST_[...] the current available tests are:
* *TEST_ADCs* - Tests training of the network created using the function GetADCPhaseNet (learning quantizers and learning samplers)
* *TEST_Quantizers* - Tests multiple quantizers for channel estimation setups.
* *TEST_ChannelEstimQuant* - Trains network for the Imperial College data.

### TEST_Quantizers

Test multiple configurations and calls different Get[...]Net functions.

### TEST_ADCs

Only the third configuration of v_nCurves is executed. There the function GetADCPhaseNet is called. This function creates a network with a layer of learning quantizers and a layer of learning samplers.
The training information is printed to the Command Window.

### TEST_ChannelEstimQuant

* Handles the data of the Imperial College (HFDL Matrixes) and trains a network using it.
* During the train, a figure is opened. This figure will record each performed test, if the script is terminated before its end, use delete_all_fig function if the figure is having a problem closing.
* The results figure, and the relevant Workspace variables will be saved to a folder named 'Results' in the working directory automatically.
