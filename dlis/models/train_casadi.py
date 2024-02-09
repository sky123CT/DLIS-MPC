import torch
import torch.nn as nn
import numpy as np
from model import CSDLIS_DF_RE, CSDLIS_MLP, DP


def training(model, inputs, labels, epoch_range=400, batch_size=16, learning_rate=0.0002):
    loss = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    costs = []
    for epoch in range(epoch_range):
        batch_costs = []
        for head in range(0, inputs.shape[0], batch_size):
            tail = head + batch_size if head + batch_size < inputs.shape[0] else inputs.shape[0]
            device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
            x = torch.tensor(inputs[head:tail, :], dtype=torch.float, requires_grad=True, device=device)
            y = torch.tensor(labels[head:tail, :], dtype=torch.float, requires_grad=True, device=device)
            predictions = model(x)
            cost = loss(predictions, y)
            optimizer.zero_grad()
            cost.backward(retain_graph=True)
            optimizer.step()
            batch_costs.append(cost.data.cpu().numpy())

        if epoch % 100 == 0:
            costs.append(np.mean(batch_costs))
            print(epoch, costs[int(epoch / 100)])
            print(batch_costs[0])

    torch.save(model.state_dict(), './model_casadi.pkl')


def main():
    torch.cuda.set_device(0)
    model = CSDLIS_MLP(mlp_i_dim=4, mlp_h_num=2, mlp_h_dim=64, mlp_act=['ReLU', 'ReLU', 'ReLU', 'ReLU']).cuda(0)
    # model = CSDLIS_DF_RE(df_i_dim=2).cuda(0)

    training_set_path = './data/dataset/train_circle_4800.xlsx'
    train_input, train_label = DP.data_processing(data_path=training_set_path, obstacle_shape='circle')
    training(model, train_input, train_label)


if __name__ == '__main__':
    main()
