import networkx as nx
import matplotlib.pyplot as plt

def build_house() -> nx.Graph:

    nodes = [
        'bottom_left', 
        'bottom_right', 
        'top_left', 
        'top_right', 
        'roof'
    ]

    edges = [
        ('bottom_left', 'top_left'), 
        ('top_left', 'top_right'),
        ('top_right', 'bottom_right'), 
        ('bottom_right', 'bottom_left'),
        ('top_left', 'roof'), 
        ('roof', 'top_right')
    ]

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph

if __name__ == '__main__':
    graph = build_house()
    nx.draw(graph)
    plt.show()
