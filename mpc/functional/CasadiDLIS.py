import torch
from casadi import *
from dlis.model_learning.model import CSDLIS_DF_RE, CSDLIS_MLP, CSDLIS_MLP2


class CasadiDLIS:
    def __init__(self, p_path='/home/ct_dual_ubuntu/Projects/hiwi-dlis/mpc-dlis/dlis/model_learning/model_casadi_2.pkl'):
        self.parameter_path = p_path
        # self.model = CSDLIS_DF_RE()
        # self.model = CSDLIS_MLP(mlp_i_dim=4, mlp_h_num=4, mlp_act=['ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU'])
        self.model = CSDLIS_MLP(mlp_i_dim=5, mlp_h_num=2, mlp_h_dim=64, mlp_act=['ReLU', 'ReLU', 'ReLU', 'ReLU'])
        self.model.load_state_dict(torch.load(self.parameter_path))

    def test_model(self):
        print(self.model)


class CasadiDLIS2Trailers:
    def __init__(self, p_path='/home/ct_dual_ubuntu/Projects/hiwi-dlis/mpc-dlis/dlis/model_learning/model_casadi_2trailer_new3.pkl'):
        self.parameter_path = p_path
        self.model = CSDLIS_MLP2(mlp_i_dim=9,
                                 mlp_h_num=2,
                                 mlp_h_dim=128,
                                 mlp_o_dim=2,
                                 mlp_act=['ReLU', 'ReLU', 'ReLU', 'ReLU'])
        self.model.load_state_dict(torch.load(self.parameter_path))
        i = MX.sym("i", 9, 1)
        model_approx_2 = self.model.approx(x=i, order=2)
        self.model_approx_2 = Function('model_approx',
                                       [i,
                                        self.model.sym_approx_params(flat=True, order=2)],
                                       [model_approx_2])
        self.model_approx_param_2 = self.model.approx_params(flat=True, order=2, a=np.zeros((9,)))
        model_approx_1 = self.model.approx(x=i, order=1)
        self.model_approx_1 = Function('model_approx',
                                       [i,
                                        self.model.sym_approx_params(flat=True, order=1)],
                                       [model_approx_1])
        self.model_approx_param_1 = self.model.approx_params(flat=True, order=1, a=np.zeros((9,)))

    def test_model(self):
        print(self.model)
