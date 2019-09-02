# What does each python file do?
1. [main.py](#main)
2. [stages.py](#stages)
3. [dataobjects.py](#dataobjects)
4. [training.py](#training)
5. [namesdict.py](#namesdict)
6. [lof.py](#lof)
7. [utils.py](#utils)

# main.py <a name="main"></a>
This is the main python script which brings the whole pipeline together (its the script that needs to be ran). The general outline is as follows:
   1) Initialise TBs and then filter them (LHCb topologoical trigger MVA)
   2) Initialise ETs and then filter them (Isolation MVA)
   2.5) Combine TBs and ETs and apply LOF calculation
   3) Feed TBs+ETs+LOF into tagger which then predicts a final output tag

## stages.py <a name="stages"></a>
This python script contains all the 'stage' functions which make up the majority of the 2B^2 tagging pipeline. Each stage has an instance of machine learning (ML or MVA) and can be summarised as the following:
   1) FirstStage is all about TwoBodys (TBs) and uses an MVA to filter the ones that don't belong to decay of the tagging B
   2) SecondStage is all ExtraTracks (ETs) and uses an MVA to filter the ones that don't belong to decay of the tagging B
   3) ThirdStage is all about the whole events themselves and using an MVA to predict the tag of signal B (but in reality
      we perform MVA on TBs and then combine the result of the TBs which are present in any given event)

## dataobjects.py <a name="dataobjects"></a>
This is the main data object to describe the TwoBodies (TBs) and the ExtraTracks (Ets).

The Object class works on 2 main inputs, the root df containing all the data, and a names dictionary which contains
the names of the branches/columns needed for each type of process/procedure.

The Object class works by only creating a df when its called upon by one of the methods, for example, when you want to apply the MVA to the TBs in the firstStage, you first initialise an instance of the object with the path to the data (same for both TBs and ETs) and the specific TB_dict (unique for TBs), then applying the get_MVAdf, the object looks up in the dictionary the columns/branches to call from the rootdf and outputs it.

The reason why a single object class was created for both the TBs and ETs was because they shared lots of similar properties and processes, for example, both TBs and ETs need to be put through MVA to filter out the bad TBs/ETs, they both also have to have their COM variables calculated via the Line of FLight technique.

Most of the objects attributes include the elements of the dictionary, which means that the object has all the info it needs to reconstruct the df4MVA or df4LOF by itself which just means that we dont have to have any extra variables, we only have to input this data object. The rest of the attributes are there mostly then to cover the subtle differences between data structures of the TBs and ETs.

## training.py <a name="training"></a>
This is the function which deals with the training and evaluation of the ML algorithms, and is currently the same for the MVA in the firstStage (TBs), the secondStage (ETs) and the thirdStage (TAGs), will create specific more tailored functions for each stage later. For both the first and second Stages, the CV function takes in the data class object twoBBdf, then using the attributes of the twoBBdf object, the data is split up into X (columns holding the features) and y (the column holding the label/target).

The data is then split into a 2:1 train test split (based on default settings governed by the test_size parameter.Then the model is trained by a nfold cross fold validation and predicted on the training data it was trained on (Out Of Fold, oof) and also predicted on the 33% of test data which the model has not seen (preds).

The results on the training and test data are evaluated and should have roughly the same performance, otherwise, it shows signs of overfitting.

The predictions by the ML algorithm (in this case its LightGBM) are calibrated into probabilities. These probabilities along with the corresponding event/TB/ET id are outputted as a panda series

## namesdict.py <a name="namesdict"></a>
These are the name dictionaries which need to be passed when initiating a twoBBdf object to specify what kind of branches/features are needed for each process. The keys represent the different type of processes which go on and include:
   1) ids
   2) MVA_key
   3) flatMVA_key
   4) label_key
   5) LOF_key
   6) flatLOF_key
The ids key needs to be specified for both the TB and ET instances, but something like the flatMVA_key can be defined as an empty list as there's no features/branches that need to be flattened when doing the TB MVA.

## lof.py <a name="lof"></a>
This python file contains all the necessary functions to calculate the Centre of Mass (COM) quantities using the Line of Flight (LOF) technique.

## utils.py <a name="utils"></a>
Python file containing specific purpose single use functions used in different part of the pipeline which I thought would be cleaner and neater to have them all stored here


