import torch
import numpy as np
import torch.nn as nn
import torch
import pandas as pd


def get_normalize(graph, normalize=True):
    """
    return the laplacian of the graph.

    :param graph: the graph structure without self loop, [N, N].
    :param normalize: whether to used the normalized laplacian.
    :return: graph laplacian.
    """
    if normalize:
        D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
        I = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
        A1 = graph + I
        L = torch.mm(torch.mm(D, A1), D)

    return L


class GF(nn.Module):
    def __init__(self):
        super(GF, self).__init__()
        self.weight = nn.Parameter(torch.randn(2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, B):
        print(self.weight)
        w1, w2 = self.sigmoid(self.weight)
        print(w1)
        print(w2)
        out=w1 * A + w2 * B
        print(out.shape)
        print(out)
        out = torch.add(w1 * A, w2 * B)
        print(out)
        print(out.shape)

if __name__ == '__main__':

    A = pd.read_csv("new.csv", header=None)
    B = pd.read_csv("correlation_all_big.csv", header=None)
    A=A.values.astype(float)
    B=B.iloc[1:,1:]
    print(B.shape)
    B=B.values.astype(float)
    A = torch.from_numpy(A)
    B = torch.from_numpy(B)
    print(A.shape)
    print(B.shape)
    A1 = get_normalize(A)
    B1 = get_normalize(B)
    net = GF()
    out = net(A1, B1)
    print(out.isnan())

    print(type(out))
