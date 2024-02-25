import torch
import torch.nn as nn
from model import DLIS, DP


def validation(model, val_path):
    loss = nn.MSELoss(reduction='mean')
    inputs, labels = DP.data_processing(data_path=val_path)
    x = torch.tensor(inputs, dtype=torch.float)
    true_results = torch.tensor(labels, dtype=torch.float)
    predictions = model(x)

    error = loss(predictions, true_results)

    for i in range(labels.shape[0]):
        print(predictions.data.numpy()[i], labels[i])

    print(error.data)


def main():
    validation_set_path = './data/dataset/val_data.xlsx'
    model = DLIS()
    model.load_state_dict(torch.load('./model.pkl'))
    validation(model, val_path=validation_set_path)


if __name__ == '__main__':
    main()
