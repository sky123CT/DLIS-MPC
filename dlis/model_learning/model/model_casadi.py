import torch
from casadi import *
from ml_casadi.torch.modules import TorchMLCasadiModule
from ml_casadi.torch.modules import nn as casadi_nn


class DLISCasadiDFRE(TorchMLCasadiModule):
    def __init__(self, df_i_dim=8,
                 df_h_num=0,
                 df_h_dim=32,
                 df_o_dim=4,
                 df_act='Sigmoid',
                 re_i_dim=6,
                 re_h_num=0,
                 re_h_dim=32,
                 re_o_dim=1,
                 re_act='ReLU'):
        super().__init__()

        # Definition of distance_feature Layers
        self.df_input_dim = df_i_dim
        self.df_hidden_num = df_h_num
        self.df_hidden_dim = df_h_dim
        self.df_output_dim = df_o_dim
        self.df_activation = df_act

        # Definition of Regression Layers
        self.re_input_dim = re_i_dim
        self.re_hidden_num = re_h_num
        self.re_hidden_dim = re_h_dim
        self.re_output_dim = re_o_dim
        self.re_activation = re_act

        # DF Layer Definition
        if self.df_activation is None:
            self.df_activation_layer = lambda x: x
        elif type(self.df_activation) is str:
            self.df_activation_layer = getattr(casadi_nn.activation, self.df_activation)()
        else:
            self.df_activation_layer = self.df_activation

        self.df_layer = []
        self.df_layer.append(casadi_nn.Linear(self.df_input_dim, self.df_hidden_dim))
        self.df_layer.append(self.df_activation_layer)
        for i in range(self.df_hidden_num):
            self.df_layer.append(casadi_nn.Linear(self.df_hidden_dim, self.df_hidden_dim))
            self.df_layer.append(self.df_activation_layer)
        self.df_layer.append(casadi_nn.Linear(self.df_hidden_dim, self.df_output_dim))
        self.df_layer.append(self.df_activation_layer)

        self.df_layers = torch.nn.ModuleList(self.df_layer)

        # RE Layer Definition
        if self.re_activation is None:
            self.re_activation_layer = lambda x: x
        elif type(self.re_activation) is str:
            self.re_activation_layer = getattr(casadi_nn.activation, self.re_activation)()
        else:
            self.re_activation_layer = self.re_activation

        self.re_layer = []
        self.re_layer.append(casadi_nn.Linear(self.re_input_dim, self.re_hidden_dim))
        self.re_layer.append(self.re_activation_layer)
        for i in range(self.re_hidden_num):
            self.re_layer.append(casadi_nn.Linear(self.re_hidden_dim, self.re_hidden_dim))
            self.re_layer.append(self.re_activation_layer)
        self.re_layer.append(casadi_nn.Linear(self.re_hidden_dim, self.re_output_dim))
        self.re_layer.append(self.re_activation_layer)

        self.re_layers = torch.nn.ModuleList(self.re_layer)

    def forward_df(self, x):
        for layer in self.df_layers:
            x = layer(x)
        out = x
        return out

    def forward_re(self, x):
        for layer in self.re_layers:
            x = layer(x)
        out = x
        return out

    def forward(self, x):
        if type(x) == SX or type(x) == MX:
            distance_feature = self.forward_df(x[:-2])
            output = self.forward_re(vertcat(distance_feature, x[-2:]))
        else:
            if x.dim() == 1:
                distance_feature = self.forward_df(x[:-2])
                output = self.forward_re(torch.cat((distance_feature, x[-2:]), 0))
            else:
                distance_feature = self.forward_df(x[:, :-2])
                output = self.forward_re(torch.cat((distance_feature, x[:, -2:].reshape(-1, 2)), 1))
        return output


class DLISCasadiMLP(TorchMLCasadiModule):
    def __init__(self, mlp_i_dim=5,
                 mlp_h_num=2,
                 mlp_h_dim=32,
                 mlp_o_dim=1,
                 mlp_act=None):
        super().__init__()

        # Definition of mlp Layers
        self.mlp_input_dim = mlp_i_dim
        self.mlp_hidden_num = mlp_h_num
        self.mlp_hidden_dim = mlp_h_dim
        self.mlp_output_dim = mlp_o_dim
        self.mlp_activation = mlp_act

        # MLP Layer Definition
        if type(self.mlp_activation) is list:
            if len(self.mlp_activation) == self.mlp_hidden_num+2:
                self.mlp_activation_layer = []
                for i in range(len(self.mlp_activation)):
                    if self.mlp_activation[i] == 'Pass':
                        self.mlp_activation_layer.append(lambda x: x)
                    else:
                        self.mlp_activation_layer.append(getattr(casadi_nn.activation, self.mlp_activation[i])())
            else:
                raise ValueError("Activation layer number can not fit Linear layer number!")
        else:
            raise ValueError("please give proper activation layer list!")

        self.mlp_layer = []
        self.mlp_layer.append(casadi_nn.Linear(self.mlp_input_dim, self.mlp_hidden_dim))
        self.mlp_layer.append(self.mlp_activation_layer[0])
        for i in range(self.mlp_hidden_num):
            self.mlp_layer.append(casadi_nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim))
            self.mlp_layer.append(self.mlp_activation_layer[i+1])
        self.mlp_layer.append(casadi_nn.Linear(self.mlp_hidden_dim, self.mlp_output_dim))
        self.mlp_layer.append(self.mlp_activation_layer[-1])

        self.mlp_layers = torch.nn.ModuleList(self.mlp_layer)

    def forward_mlp(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
        out = x
        return out

    def forward(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
        output = x
        return output


class DLISCasadiMLP2Trailers(TorchMLCasadiModule):
    def __init__(self, mlp_i_dim=9,
                 mlp_h_num=2,
                 mlp_h_dim=64,
                 mlp_o_dim=2,
                 mlp_act=None):
        super().__init__()

        # Definition of mlp Layers
        self.mlp_input_dim = mlp_i_dim
        self.mlp_hidden_num = mlp_h_num
        self.mlp_hidden_dim = mlp_h_dim
        self.mlp_output_dim = mlp_o_dim
        self.mlp_activation = mlp_act

        self.input_size = self.mlp_input_dim
        self.output_size = self.mlp_output_dim

        # MLP Layer Definition
        if type(self.mlp_activation) is list:
            if len(self.mlp_activation) == self.mlp_hidden_num+2:
                self.mlp_activation_layer = []
                for i in range(len(self.mlp_activation)):
                    if self.mlp_activation[i] == 'Pass':
                        self.mlp_activation_layer.append(lambda x: x)
                    else:
                        self.mlp_activation_layer.append(getattr(casadi_nn.activation, self.mlp_activation[i])())
            else:
                raise ValueError("Activation layer number can not fit Linear layer number!")
        else:
            raise ValueError("please give proper activation layer list!")

        self.mlp_layer = []
        self.mlp_layer.append(casadi_nn.Linear(self.mlp_input_dim, self.mlp_hidden_dim))
        self.mlp_layer.append(self.mlp_activation_layer[0])
        for i in range(self.mlp_hidden_num):
            self.mlp_layer.append(casadi_nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim))
            self.mlp_layer.append(self.mlp_activation_layer[i+1])
        self.mlp_layer.append(casadi_nn.Linear(self.mlp_hidden_dim, self.mlp_output_dim))
        self.mlp_layer.append(self.mlp_activation_layer[-1])

        self.mlp_layers = torch.nn.ModuleList(self.mlp_layer)

    def forward_mlp(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
        out = x
        return out

    def forward(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
        output = x
        return output
