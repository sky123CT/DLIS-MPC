import torch
import torch.nn as nn


class DLIntersectionModel(nn.Module):
    def __init__(self, input_dim_df=8,
                 hidden_dim_df=64,
                 output_dim_df=1,
                 input_dim_regression=2,
                 hidden_dim_regression=16,
                 output_dim_regression=1):
        super(DLIntersectionModel, self).__init__()

        self.input_dim_df = input_dim_df
        self.hidden_dim_df = hidden_dim_df
        self.output_dim_df = output_dim_df

        self.input_dim_regression = input_dim_regression
        self.hidden_dim_regression = hidden_dim_regression
        self.output_dim_regression = output_dim_regression

        self.linear_df_1 = nn.Linear(self.input_dim_df, self.hidden_dim_df)
        self.linear_df_2 = nn.Linear(self.hidden_dim_df, self.output_dim_df)
        self.activate_df = nn.Sigmoid()

        self.linear_regression_1 = nn.Linear(self.input_dim_regression, self.hidden_dim_regression)
        self.linear_regression_2 = nn.Linear(self.hidden_dim_regression, self.output_dim_regression)
        self.activate_regression = nn.ReLU()

    def forward_distance_feature(self, x):
        hidden = self.linear_df_1(x)
        activate = self.activate_df(hidden)
        out = self.linear_df_2(activate)
        return out

    def forward_regression(self, x):
        hidden1 = self.linear_regression_1(x)
        out1 = self.activate_regression(hidden1)
        hidden2 = self.linear_regression_2(out1)
        out2 = self.activate_regression(hidden2)
        return out2

    def forward(self, x):
        feature = self.forward_distance_feature(x[:, :-1])
        out = self.forward_regression(torch.cat((feature, x[:, -1].reshape(-1, 1)), 1))
        return out
