import pandas as pd
import torch
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


# metric
def metric(pred, label):
    pred = pred.clone().detach().cpu().numpy()
    label = label.clone().detach().cpu().numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(pred - label).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)

    return mae, rmse, mape


def seq2instance(data, num_his, num_pred, args=None):  # num_his = 12; num_pred = 12
    num_step, dims = data.shape  # num, 325
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, dims)
    y = torch.zeros(num_sample, num_pred, dims)
    if args:
        xd = torch.zeros(num_sample, args.num_day, dims)  # num, 3, 325
        xw = torch.zeros(num_sample, args.num_week, dims)  # to be continued
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
        temp_d = []
        temp_w = []
        for k in range(args.num_day - 1, -1, -1):
            temp_d.append(i + num_his - 1 - k * 288)

        for k in range(args.num_week - 1, -1, -1):
            temp_w.append(i + num_his - 1 - k * 7 * 288)

        for j in range(args.num_day):
            if temp_d[j]>=0:
                xd[i][j] = data[temp_d[j]]
        for j in range(args.num_week):
            if temp_w[j]>=0:
                xw[i][j] = data[temp_w[j]]

    return x, xd, xw, y


def load_data(args):
    device = args.DEVICE
    # Traffic
    df = pd.read_hdf(args.traffic_file)
    # print(df)
    traffic = torch.from_numpy(df.values)
    # print('raw data shape:', traffic.shape)
    # train/val/test
    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)  # round
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = traffic[: train_steps]
    val = traffic[train_steps: train_steps + val_steps]
    test = traffic[-test_steps:]
    trainX, trainxd, trainxw, trainY = seq2instance(train, args.num_his, args.num_pred, args)
    valX, valxd, valxw, valY = seq2instance(val, args.num_his, args.num_pred, args)
    testX, testxd, testxw, testY = seq2instance(test, args.num_his, args.num_pred, args)
    # normalization
    mean, std = torch.mean(trainX), torch.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    trainxd = (trainxd - mean) / std
    valxd = (valxd - mean) / std
    testxd = (testxd - mean) / std

    trainxw = (trainxw - mean) / std
    valxw = (valxw - mean) / std
    testxw = (testxw - mean) / std



    SE = []

    # temporal embedding
    time = pd.DatetimeIndex(df.index)
    # print(time)
    week_list = [0,1,2,3,4,5,6]
    day_list = [i for i in range(288)]
    dayofweek = []
    timeofday = []
    daycount = 0
    weekcount = 0
    for i, _ in enumerate(time):
        timeofday.append(day_list[daycount])
        daycount += 1
        if daycount == 288:
            daycount = 0

        dayofweek.append(week_list[weekcount//288])
        weekcount += 1
        if weekcount == 7*288:
            weekcount = 0

    dayofweek, timeofday = np.array(dayofweek), np.array(timeofday)
    # print(type(time))
    dayofweek = torch.reshape(torch.tensor(dayofweek), (-1, 1))
    timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
    time = torch.cat((dayofweek, timeofday), -1)
    # train/val/test
    train = time[: train_steps]
    val = time[train_steps: train_steps + val_steps]
    test = time[-test_steps:]
    # shape = (num_sample, num_his + num_pred, 2)
    trainT, _, _, trainE = seq2instance(train, args.num_his, args.num_pred, args)
    trainTE = torch.cat((trainT, trainE), 1).type(torch.int32)
    valT, _, _, valE = seq2instance(val, args.num_his, args.num_pred, args)
    valTE = torch.cat((valT,valE), 1).type(torch.int32)
    testT, _, _, testE = seq2instance(test, args.num_his, args.num_pred, args)
    testTE = torch.cat((testT,testE), 1).type(torch.int32)

    trainX, trainY, testX, testY, valX, valY = trainX.to(device), trainY.to(device), testX.to(device), testY.to(device), valX.to(device), valY.to(device)
    train_set = torch.utils.data.TensorDataset(trainX, trainxd, trainxw, trainTE, trainY)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle)

    val_set = torch.utils.data.TensorDataset(valX, valxd, valxw, valTE, valY)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    test_set = torch.utils.data.TensorDataset(testX, testxd, testxw, testTE, testY)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    ifo = {'trainshape': (trainX.shape, trainxd.shape, trainxw.shape, trainTE.shape, trainY.shape),
          'valshape':(valX.shape, valxd.shape, valxw.shape, valTE.shape, valY.shape),
           'testshape':(testX.shape, testxd.shape, testxw.shape, testTE.shape, testY.shape)}
    return (train_loader, val_loader, test_loader,
            SE, mean, std, ifo)


# dataset creation
class dataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.len = data_x.shape[0]

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.len


# statistic model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mae_loss(pred, label):
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.tensor(0.0), mask)
    loss = torch.abs(torch.sub(pred, label))
    loss *= mask
    loss = torch.where(torch.isnan(loss), torch.tensor(0.0), loss)
    loss = torch.mean(loss)
    return loss


# # plot train_val_loss
# def plot_train_val_loss(train_total_loss, val_total_loss, file_path):
#     plt.figure(figsize=(10, 5))
#     plt.plot(range(1, len(train_total_loss) + 1), train_total_loss, c='b', marker='s', label='Train')
#     plt.plot(range(1, len(val_total_loss) + 1), val_total_loss, c='r', marker='o', label='Validation')
#     plt.legend(loc='best')
#     plt.title('Train loss vs Validation loss')
#     plt.savefig(file_path)


# plot test results
def save_test_result(trainPred, trainY, valPred, valY, testPred, testY):
    with open('./figure_pems/test_results.txt', 'w+') as f:
        for l in (trainPred, trainY, valPred, valY, testPred, testY):
            f.write(list(l))

if __name__ == '__main__':
    d = 0
    # (train_loader, trainTE, val_loader, valTE, test_loader, testTE,
    #  SE, mean, std) = load_data(args)
