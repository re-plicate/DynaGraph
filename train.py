import time
import datetime
from data.utils import log_string
# from model.model_ import *
from data.utils import load_data, count_parameters
# import math
from MODEL import DynaGraph
import torch
from data.generate_adj import read_adj
import pandas as pd

def train(model, args, log, loss_criterion, optimizer, scheduler):

    (train_loader, val_loader, test_loader,
     SE, mean, std, ifo) = load_data(args)

    wait = 0
    weight_mark = 0
    val_loss_min = float('inf')
    test_loss_min = float('inf')
    best_model_wts = None
    train_total_loss = []
    val_total_loss = []
    test_total_loss = []

    for epoch in range(args.max_epoch):
        if wait >= args.patience:
            log_string(log, f'early stop at epoch: {epoch:04d}')
            break
        start_train = time.time()
        model.train()
        train_loss = 0
        num_train = 0
        for ind, data in enumerate(train_loader):
            xc, xd, xw, te, y = data  # B T N -> x need: B C N T
            xc, xd, xw = xc.unsqueeze(1).permute(0,1,3,2), xd.unsqueeze(1).permute(0,1,3,2), xw.unsqueeze(1).permute(0,1,3,2)
            optimizer.zero_grad()
            # print(xw, xd)  
            pred = model(xc, xd, xw, te)  # 32 12 325 B T N
            pred = pred * std + mean
            loss_batch = loss_criterion(pred, y)  
            num_train += xc.shape[0]
            train_loss += float(loss_batch) * xc.shape[0]
            # print(xc.shape[0])
            loss_batch.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (ind + 1) % 20 == 0:
                print(f'Training batch: {ind + 1} in epoch:{epoch}, training batch loss:{loss_batch:.4f}')
            del xc, xd, xw, y, loss_batch
        train_loss /= num_train
        train_total_loss.append(train_loss)
        end_train = time.time()


        '--------------------------------'
        ## test start
        start_test = time.time()
        test_loss = 0
        num_test = 0
        model.eval()
        with torch.no_grad():
            for ind, data in enumerate(test_loader):
                xc, xd, xw, te, y = data  # B T N -> x need: B C N T； te：B 2T 2
                xc, xd, xw = xc.unsqueeze(1).permute(0, 1, 3, 2), xd.unsqueeze(1).permute(0, 1, 3, 2), xw.unsqueeze(1).permute(0, 1, 3, 2)
                optimizer.zero_grad()
                pred = model(xc, xd, xw, te)  # 32 12 325 B T N
                pred = pred * std + mean
                loss_batch = loss_criterion(pred, y)
                test_loss += loss_batch * xc.shape[0]
                num_test += xc.shape[0]
                del xc, xd, xw, y, loss_batch
        test_loss /= num_test
        test_total_loss.append(test_loss)
        end_test = time.time()
        log_string(
            log,
            '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
                args.max_epoch, end_train - start_train, end_test - start_test))
        log_string(
            log, f'train loss: {train_loss:.4f}, test_loss: {test_loss:.4f}')
        if test_loss <= test_loss_min:
            log_string(
                log,
                f'test loss decrease from {test_loss_min:.4f} to {test_loss:.4f}, saving model to {args.model_file}')
            wait = 0
            test_loss_min = test_loss
            best_model_wts = model.state_dict()
            weight_mark = epoch
            # model.load_state_dict(best_model_wts)
            torch.save(model, './result/ST_PEMS_testbest_ez{0}_layer{1}_epoch{2}'.format(args.embed_size,args.num_layers, epoch))  #
        else:
            wait += 1


        scheduler.step()

    model.load_state_dict(best_model_wts)
    torch.save(model, args.model_file)
    log_string(log, f'Training and validation are completed, and model has been stored as {args.model_file}')
    return train_total_loss, val_total_loss

'--------------------------------------------------'


if __name__=='__main__':
    from args_parameter import *

    # log
    log = open(args.log_file, 'w')
    log_string(log, str(args)[10: -1])
    T = 24 * 60 // args.time_slot
    log_string(log, 'loading data...')
    (train_loader, trainTE, val_loader, valTE, test_loader, testTE,
     SE, mean, std, ifo) = load_data(args)

    # (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
    #  testY, SE, mean, std) = load_data(args)
    trainshape, valshape, testshape = ifo['trainshape'], ifo['valshape'], ifo['testshape']
    log_string(log, f'trainX: {trainshape[0:3]}\t\t trainY: {trainshape[-1]}')
    log_string(log, f'valX:   {valshape[0:3]}\t\tvalY:   {valshape[-1]}')
    log_string(log, f'testX:   {testshape[0:3]}\t\ttestY:   {trainshape[-1]}')
    log_string(log, f'mean:   {mean:.4f}\t\tstd:   {std:.4f}')
    log_string(log, 'data loaded!')
    del trainshape, valshape, testshape, mean, std, ifo
    log_string(log, 'compiling model...')

    # model initiation
    adj_w, adj_r = read_adj('data/Adj(PeMS).txt')  
    adj_w, adj_r = adj_w.float(), adj_r.float()
    df = pd.read_hdf('data/pems_correlation_roads.h5')
    data_val = torch.from_numpy(df.values)
    corr = torch.Tensor(data_val).float()

    in_channels = 1  # Channels of input
    # embed_size = 32 # Dimension of hidden embedding features
    embed_size = 100
    time_num = T  # 288
    num_layers = 2  # Number of ST Block
    T_dim = [args.num_his, args.num_day, args.num_week]  # Input length, should be the same as prepareData.py
    output_T_dim = args.num_pred  # Output Expected length
    heads = 2  # Number of Heads in MultiHeadAttention
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
        dropout)

    model.to(args.DEVICE)

    # loss and optimizer
    loss_criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.9)
    parameters = count_parameters(model)
    log_string(log, 'trainable parameters: {:,}'.format(parameters))


    train(model, args, log, loss_criterion, optimizer, scheduler)
