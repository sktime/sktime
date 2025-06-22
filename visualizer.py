
import networkx as nx
import matplotlib.pyplot as plt

def plot_dag(model):
    G = nx.DiGraph()
    G.add_edges_from(model.edges())
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
    plt.title("Causal DAG")
    plt.show()
