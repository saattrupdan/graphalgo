import networkx as nx
from networkx.algorithms.link_analysis import pagerank_alg as pr
import pandas as pd
from typing import Iterable, Union, Optional
from build_graph import build_house

def pagerank(
    graph: Union[nx.Graph, nx.DiGraph], 
    alpha: float = .85, 
    local_nodes: Optional[Iterable] = None,
    max_iter: int = 100) -> pd.DataFrame:
    ''' The PageRank algorithm. '''

    if local_nodes is not None:
        personalization = {node: 1. for node in local_nodes}
    else:
        personalization = None

    results = pr.pagerank(
        G = graph, 
        alpha = alpha,
        personalization = personalization,
        max_iter = max_iter
    )

    df = pd.DataFrame(results.items())
    df.columns = ['node', 'pagerank']
    df.set_index('node', inplace = True)
    df.sort_values(by = 'pagerank', ascending = False, inplace = True)
    return df

if __name__ == '__main__':
    graph = build_house()
    print(pagerank(graph))
