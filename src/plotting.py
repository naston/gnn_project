import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# Provides a plot of Graph Degree Distirbution

def plot_deg_dist(G):
    deg = np.sum(nx.adjacency_matrix(G),axis=1)
    degrees, counts = np.unique(deg, return_counts=True)

    fig = plt.plot(degrees,counts)
    return fig