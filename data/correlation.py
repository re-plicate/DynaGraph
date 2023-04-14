
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

def correla():
        df = pd.read_hdf('./PEMS08.h5')
        data_val = torch.from_numpy(df.values)
        print(data_val.shape, type(data_val), type(df))

        new_df=df.corr()
        print(new_df)
        # new_df.to_excel('correlation_all_test.xls',sheet_name='data')
        new_df.to_hdf('pems08_correlation.h5', key='datafile')
        print(new_df.shape, 'already save!')

def draw():
        import seaborn as sns
        import numpy
        #
        new_df = pd.read_hdf('./pems04_correlation.h5')
        data = new_df.values
        data2 = numpy.absolute(data)
        print(data,data2)
        print(type(data))
        data3 = data2 - data
        print(data3)
        print('drawing')
        font = {'family': 'Times New Roman',
                'size': 40,
                }
        f, ax1 = plt.subplots(figsize=(10, 10))
        plt.figure(1)

        cmap1 = sns.cubehelix_palette(start=0.5, rot=2.5, gamma=0.8, as_cmap=True)

        h1 = sns.heatmap(data2, linewidths=0.0001, ax=ax1, vmax=1, vmin=0.8,center=0.8, cmap=cmap1)
        # h1 = sns.heatmap(data3, linewidths=0.0001, ax=ax1, vmax=0.3, vmin=0, center=0.1, cmap=cmap1)
        # cmap2 = sns.cubehelix_palette(start=0, rot=2, gamma=-0.8, as_cmap=True)
        #
        # h2 = sns.heatmap(data2, linewidths=0.0001, ax=ax2, vmax=1, vmin=0, center=0.8, cmap=cmap2)

        # h=sns.heatmap(data,annot=True,ax=ax,linewidths=0,cbar=False,cmap=cmap,
        #               vmax=1, vmin=-1,square=True)  # , fmt='.1f',annot_kws={'size':1,'weight':'bold', 'color':'white'}
        # ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
        cb1 = h1.figure.colorbar(h1.collections[0])
        cb1.ax.tick_params(labelsize=16)
        ax1.tick_params(axis='x',labelsize=10)
        ax1.tick_params(axis='y',labelsize=10,labelrotation=-0)
        ax1.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax1.tick_params(which="minor", bottom=False, left=False)
        ax1.set_title('Pearson correlation coefficient',font)
        ax1.set_xlabel('Nodes of Roads',font)
        ax1.set_ylabel('Nodes of Roads',font)

        # cb2 = h2.figure_pems.colorbar(h2.collections[0])
        # cb2.ax.tick_params(labelsize=16)
        # ax2.tick_params(axis='x', labelsize=10)
        # ax2.tick_params(axis='y', labelsize=10, labelrotation=-0)
        # ax2.grid(which="minor", color="w", linestyle='-', linewidth=3)
        # ax2.tick_params(which="minor", bottom=False, left=False)
        # ax2.set_title('Pearson correlation coefficient of different Methods.', font)
        # ax2.set_xlabel('Methods', font)
        # ax2.set_ylabel('Missing ratio (%)', font)

        plt.savefig("correlation_pems_325_abs.jpg",dpi=400)
        plt.show()

if __name__ =='__main__':
        correla()
        # draw()