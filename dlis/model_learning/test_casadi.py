import torch
import torch.nn as nn
from model import CSDLIS_DF_RE, CSDLIS_MLP, DP


def validation(model, val_path):
    predictions = []
    loss = nn.MSELoss(reduction='mean')
    inputs, labels = DP.data_processing(data_path=val_path, obstacle_shape='circle')
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    x = torch.tensor(inputs, dtype=torch.float, device=device)
    true_results = torch.tensor(labels, dtype=torch.float, device=device)
    for i in range(x.shape[0]):
        predictions.append(model(x[i].squeeze()))
    predictions = torch.tensor(predictions, device=device)
    error = loss(predictions, true_results)

    for i in range(labels.shape[0]):
        print(predictions.data.cpu().numpy()[i], labels[i])

    print(error.data)


def main():
    validation_set_path = './data/dataset/val_circle_1200.xlsx'
    # model = CSDLIS_DF_RE(df_i_dim=2).cuda(0)
    # model = CSDLIS_MLP(mlp_i_dim=4, mlp_h_num=4, mlp_act=['ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU']).cuda(0)
    model = CSDLIS_MLP(mlp_i_dim=5, mlp_h_num=2, mlp_h_dim=64, mlp_act=['ReLU', 'ReLU', 'ReLU', 'ReLU']).cuda(0)
    model.load_state_dict(torch.load('./model_casadi.pkl'))
    validation(model, val_path=validation_set_path)


if __name__ == '__main__':
    main()
