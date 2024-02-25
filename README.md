This framework is used for .

There are two different ways this framework enables PyTorch models in a CasADi graph:

**Naively**, where the operations of the PyTorch model are reconstructed in the CasADi graph and the learned weights are copied over. This is limited to dense multi-layer perceptrons and can be slow for large networks as CasADi is not optimized for large matrix multiplications.

**Approximated**, where the PyTorch model is abstracted as first or second order approximation. The necessary parameters are passed to the CasADi function at every function call. This enables the use of any differentiable PyTorch module. Our paper describes how the approximation can be used to efficiently apply a learned dynamics model efficiently in an MPC setting.

## Integration with Acados
To use this framework with Acados :
- Follow the [installation instructions](https://docs.acados.org/installation/index.html).
- Install the [Python Interface](https://docs.acados.org/python_interface/index.html).
- Ensure that `LD_LIBRARY_PATH` is set correctly (`DYLD_LIBRARY_PATH`on MacOS).

An example of how a PyTorch model can be used as dynamics model in the Acados framework for Model Predictive Control can be found in `examples/mpc_mlp_cnn_example.py`.

## Examples
### Approximated
```
import ml_casadi.torch as mc
import casadi as ca
import numpy as np
import torch

size_in = 6
size_out = 3
model = mc.TorchMLCasadiModuleWrapper(
    torch_module,
    input_size=size_in,
    output_size=size_out)
    
casadi_sym_inp = ca.MX.sym('inp',size_in)
casadi_sym_out = model.approx(casadi_sym_inp, order=1)
casadi_func = ca.Function('model_approx_wrapper',
                          [casadi_sym_inp, model.sym_approx_params(order=1, flat=True)],
                          [casadi_sym_out])

inp = np.ones([1, size_in])  # torch needs batch dimension
casadi_param = model.approx_params(inp, order=1, flat=True)  # order=2
casadi_out = casadi_func(inp.transpose(-2, -1), casadi_param)   # transpose for vector rep. expected by casadi

t_out = model(torch.tensor(inp, dtype=torch.float32))

print(casadi_out)
print(t_out)
```

### Naive
