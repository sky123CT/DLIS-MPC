import torch
from dlis.models.model import CSDLIS_DF_RE, CSDLIS_MLP


class CasadiDLIS:
    def __init__(self, p_path='/home/ct_dual_ubuntu/Projects/hiwi-dlis/mpc-dlis/dlis/models/model_casadi.pkl'):
        self.parameter_path = p_path
        # self.model = CSDLIS_DF_RE()
        # self.model = CSDLIS_MLP(mlp_i_dim=4, mlp_h_num=4, mlp_act=['ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU'])
        self.model = CSDLIS_MLP(mlp_i_dim=4, mlp_h_num=2, mlp_h_dim=64, mlp_act=['ReLU', 'ReLU', 'ReLU', 'ReLU'])
        self.model.load_state_dict(torch.load(self.parameter_path))

    def test_model(self):
        print(self.model)
