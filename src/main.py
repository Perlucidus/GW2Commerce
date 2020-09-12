from model import StockTrend
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

torch.random.manual_seed(17)

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

data_path = '../data'


def sliding_windows(sequence, window_size):
    windows = []
    labels = []
    for i in range(len(sequence) - window_size - 1):
        windows.append(sequence[i:i + window_size])
        labels.append(sequence[i + window_size])
    return np.array(windows), np.array(labels)


def main():
    item = pd.read_csv(f'{data_path}/buy/item_68.csv', index_col='listing_datetime', parse_dates=['listing_datetime'])
    item.sort_values('listing_datetime', inplace=True)
    data = item['unit_price'].values
    window_size = 200

    scaler = MinMaxScaler()
    scaler.fit(data[:-window_size].reshape(-1, 1))  # Fit only on train
    data = scaler.transform(data.reshape(-1, 1)).astype(np.float32)

    sequences, labels = sliding_windows(data, window_size)
    train_size = int(len(data) * 0.7)
    data_x = Variable(torch.tensor(sequences, device=device))
    data_y = Variable(torch.tensor(labels, device=device))
    train_x = Variable(torch.tensor(sequences[:train_size], device=device))
    train_y = Variable(torch.tensor(labels[:train_size], device=device))
    test_x = Variable(torch.tensor(sequences[train_size:], device=device))
    test_y = Variable(torch.tensor(labels[train_size:], device=device))

    model = StockTrend()
    model.to(device)
    model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 200
    for epoch in range(epochs):
        pred = model(train_x)
        optimizer.zero_grad()
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()
        print(loss.item())
    model.eval()
    with torch.no_grad():
        predictions = scaler.inverse_transform(model(data_x).cpu().numpy())
    labels = scaler.inverse_transform(data_y.cpu().numpy())
    plt.axvline(x=train_size, c='r', linestyle='--')

    plt.plot(labels.reshape(-1))
    plt.plot(predictions.reshape(-1))
    plt.show()

    # x = item.index.values
    # plt.plot(x[:-window_size], train_seq)
    # plt.plot(x[-window_size:], test_seq)
    # plt.plot(x, predictions)
    # plt.plot(x, predictions2)
    # plt.show()


if __name__ == '__main__':
    main()
