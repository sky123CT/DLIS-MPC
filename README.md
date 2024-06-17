MPC-DLIS
====
Environment
----
make sure your python version >= 3.10.<br>
To install the environment, please do
```
pip install requrements.txt
```
dlis
----
This file includes the documents you need for generating datasets for training the MLP of Intersection-Regression-Model.<br>
In directory ```dlis/model_learning/data```, do 
```
python <dataset_generating_doc> <sample_number> <points_number_in_one_sample> <scale>
```
"scale" is the standard deviation of the obstacle-edge around the trailer, the dataset will be saved in directory 
```dlis/model_learning/data/dataset```with the sample number, points number and scale you set<br><br>
To train the model, please go to dlis/model_learning/ and use 
```train_casadi.py```
remember to change your dataset_name for training and your model name<br>
use ```test_casadi.py```for accuracy testing, remember to change your dataset_name for testing and your model name<br>

ml-casadi
----
Framework from TUM, used for combining PyTorch and Casadi

mpc
----
main directory for MPC process<br>
for setting up `Cost Function`, go to ```mpc/functional/```and find the proper document with `cost_function_XXXX.py` for
the cost function<br>
for setting up `Constrains`, go to ```mpc/functional/```and find the proper document with `constraints_XXXX.py` for
the cost function<br>
for setting up `DLIS_Model`, go to ```mpc/functional/```and find the document with `CasadiDLIS.py` for
the Model you want to use, remember to change the model location and name<br>
for starting simulation, go to main directory `mpc-dlis/` and select the proper document for your experiment 
with `mpc_dlis_XXXX.py`