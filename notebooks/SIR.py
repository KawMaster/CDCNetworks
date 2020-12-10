# %% [markdown]
# # Spread of Epidemics - SIR Model
# %% [markdown]
# This notebook goes over Section 5 of the final project.  It simulates the SIR model on the network.
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
import random
import logging

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize': (10, 6)})
sns.set(rc={'figure.dpi': 500})
sns.set_style("white")

logging.basicConfig(filename='SIR.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
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

# %% SIR Model


def SIR(A, β, μ, Δt, T, v, V):
    '''
    Simulates the SIR model with the following parameters for a given Adjacency Matrix.

    Parameters
    ----------
        A: nx.graph, Adjacency matrix
        β: float, Infection Rate
        μ: float, Recovery Rate
        Δt: float, Time Step
        T: float, Total Simulation Time
        v: int, Patient Zero
        V: List of Susceptible Nodes
    '''
    A_np = nx.to_numpy_array(A)
    print(β)

    # Calculated Parameters
    n_time_steps = int(T / Δt)
    q = μ * Δt

    # Sets of Nodes
    susceptibleNodes = V
    infectiousNodes = [v]
    recoveredNodes = []

    iteration = 0
    logging.debug('Patient Zero Node: {}'.format(v))

    while len(infectiousNodes) != 0:

        for v in infectiousNodes:

            neighbors = [n for n in A.neighbors(v)]

            # Infected node infects it's neighbors

            for u in neighbors:
                if u in susceptibleNodes:
                    p = β * A_np[u, v] * Δt
                    print(p)
                    if np.random.uniform() < p:
                        logging.debug('----- Node {} Infected -----'.format(u))

                        infectiousNodes.append(u)
                        susceptibleNodes.remove(u)

            # Infected node can recover
            if np.random.uniform() < q:
                logging.debug('----- Node {} Recovered -----'.format(v))

                recoveredNodes.append(v)
                infectiousNodes.remove(v)
        iteration += 1

    num_recovered = len(recoveredNodes)
    num_susceptible = len(susceptibleNodes)
    return (num_recovered, num_susceptible)


# %%
# A = NETWORKS[NETWORK_KEYS[0]]
# β = 4e-4
# k = [1, 2, 3, 4, 5]
# μ = 100 * (β / k[4])
# Δt = 1e-3 * (1 / β)
# T = 1000
# v = random.sample(A.nodes(), 1)[0]
# V = [n for n in A.nodes() if n != v]


# SIR(A, β, μ, Δt, T, v, V)
# %%

# Run the simulation on the Contact Networks
R = dict()
S = dict()
Ρ = dict()

# Loop over the contact networks
for nk in NETWORK_KEYS:

    if 'pres_' not in nk:

        A = NETWORKS[nk]
        β = 4e-4
        k = [1, 2, 3, 4, 5]
        Δt = 1e-3 * (1 / β)
        T = 1000
        v = random.sample(A.nodes(), 1)[0]
        V = [n for n in A.nodes() if n != v]
        degrees = [A.degree(n) for n in A.nodes()]
        d = sum(degrees) / len(degrees)
        ρ = None

        R[nk] = dict()
        S[nk] = dict()
        Ρ[nk] = dict()

        # Looping for every value of k
        for kv in k:
            R[nk][kv] = []
            S[nk][kv] = []
            Ρ[nk][kv] = []
            μ = 100 * (β / kv)
            mu = μ

            logging.debug('Network: {} \t k: {} \t mu: {} \t v: {}'.format(nk, k, mu, v))
            # 100 trials per μ
            for t in range(100):
                logging.debug('Simulation #{}'.format(t))
                r, s = SIR(A, β, μ, Δt, T, v, V)
                R[nk][kv].append(r)
                S[nk][kv].append(s)

            ρ = (β / μ) * d  # ρ₀
            Ρ[nk][kv] = ρ


# %%

def hist_plots(data, bins, network_name):

    # Global Plotting Settings (re-executing them just in case)
    sns.set(rc={'figure.figsize': (10, 6)})
    sns.set(rc={'figure.dpi': 500})
    sns.set_style("white")

    ax = sns.displot(data=data, bins=bins, stat='probability')

    labels = {}

    labels['xlabel'] = r'Recovered Nodes'
    labels['ylabel'] = r'Probability'
    labels['title'] = 'Distribution of Recovered Nodes: {} Network'.format(
        network_name)

    ax.set(**labels)
    plt.show()
    return ax


# %%
for nk in NETWORK_KEYS:
    if 'pres_' not in nk:
        recovered = R[nk]
        rho = Ρ[nk]

        for k in range(1, 6):
            fig = hist_plots(recovered[k], 30, nk + '_k=' + str(k))


# %%
