import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def dataframe_rec(testpre, testy):
    con = np.concatenate([testpre, testy], axis=1)  # num 12+12 325

    for i in range(con.shape[-1]):
        temp = con[:,:,i].reshape(con.shape[0],con.shape[1])
        print(temp.shape)
        temp = pd.DataFrame(temp)
        temp.to_hdf('./record/node{0}_result.h5'.format(i), key='datafile', mode='w')



def visual(datapath):
    data = pd.read_hdf(datapath).values
    print(data.shape, 'datashape')
    ground_truth = np.concatenate([data[:, 12], data[-1, 13:]], axis=0)  # num+11
    pre_15 = data[:, 2]
    pre_30 = data[:, 5]
    pre_60 = data[:, 11]

    # 2017 1 1 - 2017 1 3
    # ground_truth = ground_truth[0:600]
    # pre_15 = pre_15[0:600]
    # pre_30 = pre_30[0:600]
    # pre_60 = pre_60[0:600]

    # 1.31 8:00 - 2.2 8:00
    ground_truth = ground_truth[8928+96:8928+600+96]
    pre_15 = pre_15[8928+96:8928+600+96]
    pre_30 = pre_30[8928+96:8928+600+96]
    pre_60 = pre_60[8928-11+96:8928+600-11+96]

    plt.figure(figsize=(14, 7), dpi=500)
    plt.plot(range(1, len(ground_truth) + 1), ground_truth, c='royalblue', marker='|', label='Ground truth')
    plt.plot(range(1, len(pre_15) + 1), pre_15, c='gray', linestyle='-', marker='+', label='15 min',zorder = 0)
    plt.plot(range(1, len(pre_30) + 1), pre_30, c='g', marker=',', label='30 min')
    plt.plot(range(1, len(pre_60) + 1), pre_60, c='coral', marker=',', label='1 hour')
    plt.legend(loc='best', fontsize=14)
    plt.grid(c='lightgray', zorder=0, linestyle='--')
    # plt.title('pred and ground')
    # 1.1 00:00 - 1.3 00:00
    # plt.xticks([0, 144, 288, 432, 576], ['2017-01-01, \n00:00', '2017-01-01, \n12:00', '2017-01-02, \n00:00', '2017-01-02, \n12:00', '2017-01-03, \n00:00'])

    #1.31 8:00 - 2.2 8:00
    plt.xticks([0, 144, 288, 432, 576], ['2017-01-31, \n08:00', '2017-01-31, \n20:00', '2017-02-01, \n08:00', '2017-02-01, \n20:00', '2017-02-02, \n08:00'], fontsize= 16)
    plt.yticks([30,50,70],fontsize= 16)
    plt.ylabel('Speed (miles)', fontsize= 20)
    # plt.xlabel('Time', fontsize=16)
    plt.savefig('./record/figure/plt')
    # plt.show()


import seaborn as sns

# 以下是相关性的热力图，方便肉眼看
def show_heatMap():
    new_df = pd.read_hdf('./data/pems_correlation_roads.h5')
    dfData = new_df.values
    dfData = dfData[:20, :20]
    print(dfData[15, 1], dfData[15, 5], dfData[15, 18], dfData[1,5], dfData[1,18], dfData[5,18])
    plt.subplots(figsize=(9, 9),dpi=500)
    sns.heatmap(dfData, vmax=1, square=True, cmap="Blues")
    plt.savefig('./record/figure/cor')
    # plt.show()

def compare_node(path_ind):
    ground_list = []
    for datapath in path_ind:
        data = pd.read_hdf('./record/node{0}_result.h5'.format(datapath)).values
        # print(data.shape, 'datashape')
        ground_truth = data[:,12]
        ground_truth = ground_truth[8928 + 96:8928 + 600 + 96]
        ground_list.append(ground_truth)

    plt.figure(figsize=(13, 7), dpi=800)
    plt.plot(range(1, len(ground_list[0]) + 1), ground_list[0], c='darkblue', marker=',', label='Node {0}'.format(path_ind[0]))
    plt.plot(range(1, len(ground_list[1]) + 1), ground_list[1], c='darkgreen', marker=',', label='Node {0}'.format(path_ind[1]), zorder=0)
    plt.plot(range(1, len(ground_list[2]) + 1), ground_list[2], c='darkred', marker=',', label='Node {0}'.format(path_ind[2]))
    plt.plot(range(1, len(ground_list[3]) + 1), ground_list[3], c='goldenrod', marker=',', label='Node {0}'.format(path_ind[3]))
    plt.legend(loc='best', fontsize=14)
    plt.grid(c='lightgray', zorder=0, linestyle='--')
    # plt.title('pred and ground')
    # 1.1 00:00 - 1.3 00:00
    # plt.xticks([0, 144, 288, 432, 576], ['2017-01-01, \n00:00', '2017-01-01, \n12:00', '2017-01-02, \n00:00', '2017-01-02, \n12:00', '2017-01-03, \n00:00'])

    # 1.31 8:00 - 2.2 8:00
    plt.xticks([0, 144, 288, 432, 576],
               ['2017-01-31, \n08:00', '2017-01-31, \n20:00', '2017-02-01, \n08:00', '2017-02-01, \n20:00',
                '2017-02-02, \n08:00'], fontsize=18)
    plt.yticks(fontsize=16)
    plt.yticks([30,50,70],fontsize= 16)
    plt.ylabel('Speed (miles)', fontsize= 20)
    plt.savefig('./record/figure/compare')


if __name__ == '__main__':
    # node's prediction vs target
    # visual('./record/node25_result.h5')
    # show_heatMap()
    compare_node([1,5,15,18])