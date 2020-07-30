import networkx as nx

def build_house() -> nx.Graph:
    nodes = [1, 2, 3, 4, 5]
    edges = [(1, 2), (2, 3), (3, 4), (4, 1), (2, 5), (5, 3)]
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph

if __name__ == '__main__':
    graph = build_house()
