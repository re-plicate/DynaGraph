import torch
import time
import math
import numpy as np
from data.utils import log_string, metric
from data.utils import load_data
# import math
from MODEL import DynaGraph
import torch
from data.generate_adj import read_adj
import pandas as pd
from args_parameter import *
# from print_utils import dataframe_rec

def test(args, model, log):
    (train_loader, val_loader, test_loader,
     SE, mean, std, ifo) = load_data(args)
    model_path = args.model_file
    model = torch.load(model_path)

    log_string(log, '**** testing model ****')
    log_string(log, 'loading model from %s' % model_path)
    log_string(log, 'model restored!')
    log_string(log, 'evaluating...')

    with torch.no_grad():

        '''trainset evaluate'''
        trainPred = []
        trainY = []
        for ind, data in enumerate(train_loader):
            xc, xd, xw, te, y = data  # B T N -> x need: B C N T
            xc, xd, xw = xc.unsqueeze(1).permute(0, 1, 3, 2), xd.unsqueeze(1).permute(0, 1, 3, 2), xw.unsqueeze(1).permute(0, 1, 3, 2)

            pred_batch = model(xc, xd, xw, te)  # 32 12 325 B T N
            trainPred.append(pred_batch.to('cpu').detach().clone())
            trainY.append(y.to('cpu').detach().clone())
            if ind%100==0:
                print('trainset: %.2f%% percent is calculated' % (ind * args.batch_size/36458*100))

            del xc, xd, xw, pred_batch, y
        trainPred = torch.from_numpy(np.concatenate(trainPred, axis=0))
        trainPred = trainPred * std + mean
        trainY = torch.from_numpy(np.concatenate(trainY, axis=0))

        '''valset evaluate'''
        valPred = []
        valY = []
        for ind, data in enumerate(val_loader):
            xc, xd, xw, te, y = data  # B T N -> x need: B C N T
            xc, xd, xw = xc.unsqueeze(1).permute(0, 1, 3, 2), xd.unsqueeze(1).permute(0, 1, 3, 2), xw.unsqueeze(
                1).permute(0, 1, 3, 2)

            pred_batch = model(xc, xd, xw, te)
            valPred.append(pred_batch.to('cpu').detach().clone())
            valY.append(y.to('cpu').detach().clone())
            if ind%100==0:
                print('valset:  %.2f%% percent is calculated' % (ind * args.batch_size/5189*100))
            del xc, xd, xw, pred_batch, y
        valPred = torch.from_numpy(np.concatenate(valPred, axis=0))
        valPred = valPred * std + mean
        valY = torch.from_numpy(np.concatenate(valY, axis=0))


        '''testset evaluate'''
        testPred = []
        testY = []
        start_test = time.time()
        for ind, data in enumerate(test_loader):
            xc, xd, xw, te, y = data  # B T N -> x need: B C N T
            xc, xd, xw = xc.unsqueeze(1).permute(0, 1, 3, 2), xd.unsqueeze(1).permute(0, 1, 3, 2), xw.unsqueeze(
                1).permute(0, 1, 3, 2)

            pred_batch = model(xc, xd, xw, te)
            testPred.append(pred_batch.to('cpu').detach().clone())
            testY.append(y.to('cpu').detach().clone())
            if ind%100==0:
                print('testset:  %.2f%% percent is calculated' % (ind * args.batch_size/10400*100))
            del xc, xd, xw, pred_batch, y
        testPred = torch.from_numpy(np.concatenate(testPred, axis=0))
        testPred = testPred * std + mean
        testY = torch.from_numpy(np.concatenate(testY, axis=0))

    end_test = time.time()
    train_mae, train_rmse, train_mape = metric(trainPred, trainY)
    val_mae, val_rmse, val_mape = metric(valPred, valY)
    test_mae, test_rmse, test_mape = metric(testPred, testY)
    log_string(log, 'testing time: %.1fs' % (end_test - start_test))
    log_string(log, '                MAE\t\tRMSE\t\tMAPE')
    log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' %
               (train_mae, train_rmse, train_mape * 100))
    log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
               (val_mae, val_rmse, val_mape * 100))
    log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
               (test_mae, test_rmse, test_mape * 100))
    log_string(log, 'performance in each prediction step')
    MAE, RMSE, MAPE = [], [], []

    for step in range(args.num_pred):
        mae, rmse, mape = metric(testPred[:, step], testY[:, step])
        MAE.append(mae)
        RMSE.append(rmse)
        MAPE.append(mape)
        log_string(log, 'step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
                   (step + 1, mae, rmse, mape * 100))
    average_mae = np.mean(MAE)
    average_rmse = np.mean(RMSE)
    average_mape = np.mean(MAPE)
    log_string(
        log, 'average:         %.2f\t\t%.2f\t\t%.2f%%' %
             (average_mae, average_rmse, average_mape * 100))

    # result record on h5
    # dataframe_rec(testPred, testY)
    print('each road result has been recorded！')
    return trainPred, valPred, testPred, trainY, valY, testY

if __name__ =='__main__':
    # model initiation
    adj_w, adj_r = read_adj('data/Adj(PeMS).txt')
    adj_w, adj_r = adj_w.float(), adj_r.float()
    df = pd.read_hdf('data/pems_correlation_roads.h5')
    data_val = torch.from_numpy(df.values)
    corr = torch.Tensor(data_val).float()
    T = 24 * 60 // args.time_slot
    in_channels = 1  # Channels of input
    # embed_size = 32 # Dimension of hidden embedding features
    embed_size = args.embed_size
    time_num = T  # 288
    num_layers = args.num_layers  # Number of ST Block
    T_dim = [args.num_his, args.num_day, args.num_week]  # Input length, should be the same as prepareData.py
    output_T_dim = args.num_pred  # Output Expected length
    heads = args.heads  # Number of Heads in MultiHeadAttention
    cheb_K = 2  # Order for Chebyshev Polynomials (Eq 2)
    forward_expansion = 4  # Dimension of Feed Forward Network: embed_size --> embed_size * forward_expansion --> embed_size
    dropout = 0
    extra_feature = torch.Tensor(120, 2)
    model = DynaGraph(
        adj_r,
        corr,
        extra_feature,
        in_channels,
        embed_size,
        time_num,
        num_layers,
        T_dim,
        output_T_dim,
        heads,
        cheb_K,
        forward_expansion,
        args.DEVICE,
        dropout)

    model.to(args.DEVICE)
    log = open('data/test_log_L1loss_', 'w')
    trainPred, valPred, testPred, trainY, valY, testY = test(args, model, log)