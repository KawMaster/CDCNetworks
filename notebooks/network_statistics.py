# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Comparing the Networks using Local and Global Statistics
# %% [markdown]
# This notebook goes over Section 4 of the final project.  It calculates the $6$ statistics needed.
# %% [markdown]
# ## Libraries

# %%
# Network stuff

# Data Science
import numpy as np
import karateclub as kc
import networkx as nx


# Utilities
import pandas as pd
from tqdm.notebook import tqdm, trange
import os
import glob
import collections

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc = {'figure.figsize':(10,6)})
sns.set(rc = {'figure.dpi': 500})
sns.set_style("white")
# %% [markdown]
# ## Loading the Data

# %%
NETWORKS = {}

for filename in glob.glob('../data/processed/*.csv'):
    with open(os.path.join('.', filename), 'r') as f:
        network = pd.read_csv(f).to_numpy()
        fn = filename.split('../data/processed\\A_')[1].split('.csv')[0]
        NETWORKS[fn] = nx.from_numpy_matrix(network)
        
NETWORK_KEYS = list(NETWORKS.keys())      

# %% [markdown]

# ## Creating Network Statistics Class
# Handles all the network statistic calls
# %%
class NetworkStatistics():
    """
    Provides basic utilities to explore the statistics of the Networks.
    """
    def density(self, G):
        '''
        Parameters:
        -----------
            G: nx.graph

        Returns:
        --------
            Graph Density of the input network G
        '''
        return nx.density(G)
    
    def λ_max(self, G):
        '''
        Parameters:
        -----------
            G: nx.graph

        Returns:
        --------
            max{|λ₁|, ..., |λₙ|}: The eigenvalue with the largest magnitude
        '''
        G_temp = nx.to_numpy_array(G)
        w, v = np.linalg.eig(G_temp)
        e = np.absolute(w)
        return np.max(e)

    def deg_dist(self, G):
        '''
        Parameters:
        -----------
            G: nx.graph

        Returns:
        --------
            A dictionary of the degree and it's associated frequency.
        '''
        degree_sequence = sorted([d for n, d in G.degree()], reverse = True)
        degree_count = collections.Counter(degree_sequence)
        return degree_sequence

    def cluster_coeff(self, G):
        '''
        Parameters:
        -----------
            G: nx.graph

        Returns:
        --------
            A list of the clustering coefficients for each node in the input Graph G.
        '''
        cc = nx.clustering(G).values()
        cc_sequence = sorted(list(cc), reverse = True)
        return cc_sequence
    
    def transitivity(self, G):
        '''
        Parameters:
        -----------
            G: nx.graph

        Returns:
        --------
            The transitivity of the input Graph G.
        '''
        return nx.transitivity(G)

    def sixth_statistic(self, G):
        '''
        ??? what is the 6th stat?
        '''
        pass

# %% Network statistics Test Area
ns = NetworkStatistics()
ns.density(NETWORKS[NETWORK_KEYS[0]])
print(ns.λ_max(NETWORKS[NETWORK_KEYS[4]]))
ns.deg_dist(NETWORKS[NETWORK_KEYS[0]])

print(NETWORK_KEYS)
# %% Plotting Test

dd = ns.deg_dist(NETWORKS[NETWORK_KEYS[0]])

ax = sns.displot(data = dd, bins = 30, stat = 'probability')
ax.set(xlabel = r'Degree $d$', ylabel = 'Probability', title = 'Normalized Degree Distribution Histogram of the {} Network'.format(NETWORK_KEYS[0]))

plt.show()
# %%
def hist_plots(data, bins, network_name, plot_type = None):

    # Global Plotting Settings (re-executing them just in case)
    sns.set(rc = {'figure.figsize': (10,6)})
    sns.set(rc = {'figure.dpi': 500})
    sns.set_style("white")


    ax = sns.displot(data = data, bins = bins, stat = 'probability')

    labels = {}

    if 'pres_' in network_name:
        network_name = 'Co-Presence ' + network_name.split('pres_')[1]
        print(network_name)
    if plot_type == 'deg_dist':
        labels['xlabel'] = r'Degree $d$'
        labels['ylabel'] = r'Probability'
        labels['title'] = 'Normalized Degree Distribution: {} Network'.format(network_name)

    if plot_type == 'cluster_coeff':
        labels['xlabel'] = r'Clustering Coefficient $cc(v)$'
        labels['ylabel'] = r'Probability'
        labels['title'] = 'Normalized Clustering Coefficient Distribution: {} Network'.format(network_name)

    ax.set(**labels)
    return ax


# %% Calculate the Network Statistics

ns = NetworkStatistics()

data_dict = {'d': [], 'λ': [], 't': []}

for name in tqdm(NETWORK_KEYS):

    graph = NETWORKS[name]

    # Network Statistics
    d = ns.density(graph)
    λ = ns.λ_max(graph)
    t = ns.transitivity(graph)

    data_dict['d'].append(d)
    data_dict['λ'].append(λ)
    data_dict['t'].append(t)

data_df = pd.DataFrame.from_dict(data_dict)
data_df.to_csv('../data/results/NetworkStatistics.csv', index = False)

# %% Calculate Network Degree Distribution and Cluster Coefficients

for name in tqdm(NETWORK_KEYS):
    graph = NETWORKS[name]

    dd = ns.deg_dist(graph)
    cc = ns.cluster_coeff(graph)

    fig1 = hist_plots(data = dd, bins = 30, network_name = name, plot_type = 'deg_dist')
    fig1.savefig('../figs/' + name + '_deg_dist' + '.png')

    fig2 = hist_plots(data = cc, bins = 30, network_name = name, plot_type = 'cluster_coeff')
    fig2.savefig('../figs/' + name + '_cluster_coeff' + '.png')
# %%
