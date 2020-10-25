import numpy as np
import itertools as it
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.style.use('ggplot')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

import dgl
from dgl.nn.pytorch import GraphConv, GINConv, GATConv


def build_karate_club_graph() -> dgl.DGLGraph:
    ''' Build the famous karate club graph. '''

    # All 78 edges are stored in two numpy arrays. One for source endpoints
    # while the other for destination endpoints.
    src = np.array(
        [1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
        10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
        25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 33, 33, 33, 33]
    )
    dst = np.array(
        [0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
        5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
        24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
        29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
        31, 32]
    )

    # Edges are directional in DGL; Make them bi-directional.
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])

    # Construct a DGLGraph
    return dgl.graph((u, v))

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.lin2(x)
        return x

class GCN(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.conv1 = GINConv(MLP(emb_dim, hidden_dim, hidden_dim), 'sum')
        self.conv2 = GINConv(MLP(hidden_dim, hidden_dim, num_classes), 'sum')

    def forward(self, graph):
        embeddings = graph.ndata['embeddings']
        h0 = self.conv1(graph, embeddings)
        h = torch.relu(h0)
        h = self.conv2(graph, h)
        return h, h0

def train_net(graph, net, epochs: int, lr: float, make_animation: bool=False):
    nnodes = graph.number_of_nodes()
    embed = nn.Embedding(nnodes, net.emb_dim)
    graph.ndata['embeddings'] = torch.ones_like(embed.weight)

    labelled_nodes = torch.tensor([15, 16, 0, 2])
    labels = torch.tensor([0, 0, 1, 1])

    params = it.chain(net.parameters(), embed.parameters())
    optim = torch.optim.AdamW(params=params, lr=lr)

    all_logits = np.empty((epochs, nnodes, net.num_classes))
    all_embeds = np.empty((epochs, nnodes, net.hidden_dim))
    for epoch in range(epochs):
        optim.zero_grad()

        logits, embeds = net(graph)
        log_probs = F.log_softmax(logits, 1)
        loss = F.nll_loss(log_probs[labelled_nodes], labels)

        loss.backward()
        optim.step()

        all_logits[epoch] = logits.detach().numpy()
        all_embeds[epoch] = embeds.detach().numpy()
       
        if epoch % 100 == 0:
            print(f'Epoch {epoch} | Loss {loss.item():.4f}')


    #=== Make animation ===
    if make_animation:
        nx_graph = graph.to_networkx().to_undirected()
        layout = nx.spring_layout(nx_graph)
        colours = ['#00FFFF', '#FF00FF']

        def draw(i: int):
            logits = all_logits[i]
            embeds = all_embeds[i]
            classes = logits.argmax(-1)
            ncolours = [colours[cls] for cls in classes]

            axes[0].cla()
            axes[0].axis('off')
            axes[0].set_title(f'Epoch {i}')
            nx.draw_networkx(nx_graph, pos=layout,
                    node_color=ncolours, with_labels=True, node_size=300, ax=axes[0])

            axes[1].cla()
            axes[1].axis('off')
            axes[1].set_title(f'Epoch {i}')
            axes[1].scatter(embeds[:, 0], embeds[:, 1], c=ncolours)

        fig = plt.figure(dpi=100)
        axes = fig.subplots(1, 2)
        ani = animation.FuncAnimation(fig, draw, frames=range(0, epochs, 50), 
                                      interval=100)
        ani.save('animation.mp4')


if __name__ == '__main__':
    graph = build_karate_club_graph()
    net = GCN(emb_dim=2, hidden_dim=2, num_classes=2)
    train_net(graph, net, epochs=5000, lr=3e-3, make_animation=True)
