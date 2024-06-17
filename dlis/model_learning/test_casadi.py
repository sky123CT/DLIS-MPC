import torch
import torch.nn as nn
import numpy as np
from model import CSDLIS_DF_RE, CSDLIS_MLP, DP


def validation(model, val_path):
    predictions = []
    loss = nn.MSELoss(reduction='mean')
    # inputs, labels = DP.data_processing(data_path=val_path, obstacle_shape='circle')
    inputs, labels = DP.data_processing_2trailer(data_path=val_path)
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    x = torch.tensor(inputs, dtype=torch.float, device=device)
    true_results = torch.tensor(labels, dtype=torch.float, device=device)
    for i in range(x.shape[0]):
        predictions.append(model(x[i].squeeze()).cpu().detach().numpy())
    predictions = torch.tensor(predictions, device=device)
    error = loss(predictions.cpu(), true_results.cpu())

    for i in range(labels.shape[0]):
        print(predictions.data.cpu().numpy()[i], labels[i])

    print(error.data)


def main():
    validation_set_path = './data/dataset/sample_2trailers_200samples_with_tractor_1000points.xlsx'
    # model = CSDLIS_DF_RE(df_i_dim=2).cuda(0)
    # model = CSDLIS_MLP(mlp_i_dim=4, mlp_h_num=4, mlp_act=['ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU']).cuda(0)
    model = CSDLIS_MLP(mlp_i_dim=9,
                       mlp_h_num=2,
                       mlp_h_dim=128,
                       mlp_o_dim=2,
                       mlp_act=['ReLU', 'ReLU', 'ReLU', 'ReLU']).cuda(0)
    model.load_state_dict(torch.load('./model_casadi_2trailer_new5.pkl'))
    validation(model, val_path=validation_set_path)


if __name__ == '__main__':
    main()
