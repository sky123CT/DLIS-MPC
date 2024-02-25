# DLIS-MPC
## Structure
### dlis: model training documents
```
dlis/model_learning/train_casadi: configuration of the model structure and training.
dlis/model_learning/test_casadi: testing of the model accuraccy.
```
### ml-casadi: This framework enables trained PyTorch Models to be used in CasADi graphs and subsequently in Acados optimal control problems.
### mpc: MPC class including dynamic model, cost function, constraints etc.,  for the controlling problem
## Experiment
### DLIS-MPC
```
mpc_dlis.py
```
### Experiments with further obstacles setting
```
test_2_obstacle 
```
