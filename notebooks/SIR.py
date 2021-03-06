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
    # print(β)

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
                    # print(p)
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
        Δt = 1e-2 * (1 / β)
        T = 1000
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

            # 100 trials per μ
            for t in range(100):

                # Select a random node
                v = random.sample(A.nodes(), 1)[0]
                V = [n for n in A.nodes() if n != v]
                logging.debug(
                    'Network: {} \t k: {} \t mu: {} \t v: {}'.format(nk, k, mu, v))

                logging.debug('Simulation #{}'.format(t))
                r, s = SIR(A, β, μ, Δt, T, v, V)
                R[nk][kv].append(r)
                S[nk][kv].append(s)

            ρ = (β / μ) * d  # ρ₀
            Ρ[nk][kv] = ρ


# %%

def hist_plots(data, network_name):

    # Global Plotting Settings (re-executing them just in case)
    sns.set(rc={'figure.figsize': (10, 6)})
    sns.set(rc={'figure.dpi': 500})
    sns.set_style("white")

    meanprops = {'marker': 'o', 'markerfacecolor': 'white',
                 'markeredgecolor': 'black', 'markersize': '5'}
    # ax = sns.boxplot(x='rho', y='recovered', data=data, hue='k', dodge=False,
    #                  showmeans=True, meanprops=meanprops, boxprops=dict(alpha=.3))

    labels = dict()

    # labels['xlabel'] = r'$\rho_0$'
    # labels['ylabel'] = r'Recovered Nodes'
    # labels['title'] = 'Distribution of Recovered Nodes: {} Network'.format(network_name)

    # labels['xlabel'] = r'$\rho_0$'
    # labels['ylabel'] = r'Recovered Nodes Fraction $n_r$'
    # labels['title'] = r'Distribution of the Recovered Nodes Fraction $n_r$: {} Network'.format(
    #     network_name)

    labels['xlabel'] = r'$\rho_0$'
    labels['ylabel'] = r'Recovered Nodes'
    labels['title'] = 'Distribution of Recovered Nodes with Averages: {} Network'.format(
        network_name)

    ax = sns.boxplot(x='rho', y='recovered', data=data, dodge=False,
                     showmeans=True, meanprops=meanprops, boxprops=dict(alpha=.3))

    ax.set(**labels)

    # Ticks

    ticks = ax.get_xticklabels()
    ticks = [ρ.get_text() for ρ in ticks]
    ticks = [round(float(ρ), 4) for ρ in ticks]
    ax.set_xticklabels(ticks)

    # Labeling Means

    means = [round(m, 4) for m in data.groupby(
        ['rho'])['recovered'].mean().to_list()]
    print(type(means))

    vertical_offset = data['recovered'].mean() * -0.05

    for xtick in ax.get_xticks():
        ax.text(xtick, means[xtick] + vertical_offset, r'$\bar{{n}}_r = {0}$'.format(means[xtick]),
                horizontalalignment='center', size='x-small', color='black', weight='semibold', bbox=dict(facecolor='#FFFFFF', alpha = 1))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='k')
    ax.legend(loc='upper left')
    ax.get_legend().set_title("k")
    plt.show()

    return ax.get_figure()


# %% 5.2.4
for nk in NETWORK_KEYS:
    if 'pres_' not in nk:
        recovered = R[nk]
        rho = Ρ[nk]

        data = {}
        data['recovered'] = []
        data['rho'] = []
        data['k'] = []
        for k in range(1, 6):
            reck = recovered[k]
            rhok = rho[k]

            data['recovered'].extend(reck)
            data['rho'].extend([rhok for _ in reck])
            data['k'].extend([k for _ in reck])

        df = pd.DataFrame(data)
        fig = hist_plots(df, nk)
        fig.savefig('../figs/recovered_nodes/' + nk + '_deg_dist' + '.png')


# %% 5.2.5

for nk in NETWORK_KEYS:
    if 'pres_' not in nk:
        A = NETWORKS[nk]
        num_nodes = A.number_of_nodes()

        recovered = R[nk]
        rho = Ρ[nk]

        data = {}
        data['recovered'] = []
        data['rho'] = []
        data['k'] = []
        for k in range(1, 6):
            reck = recovered[k]
            reck_frac = [r / num_nodes for r in reck]
            reck_20 = [r for r in reck_frac if r > 0.2]  # this is the nr value
            rhok = rho[k]

            data['recovered'].extend(reck_20)
            data['rho'].extend([rhok for _ in reck_20])
            data['k'].extend([k for _ in reck_20])

        df = pd.DataFrame(data)
        fig = hist_plots(df, nk)
        fig.savefig('../figs/recovered_nodes/' + nk + '_nr' + '.png')

# %% 5.2.6

for nk in NETWORK_KEYS:
    if 'pres_' not in nk:
        A = NETWORKS[nk]
        num_nodes = A.number_of_nodes()

        recovered = R[nk]
        rho = Ρ[nk]

        data = {}
        data['recovered'] = []
        data['rho'] = []
        data['k'] = []
        for k in range(1, 6):
            reck = recovered[k]
            reck_20 = [r for r in reck if r > 0.2*num_nodes]
            rhok = rho[k]

            data['recovered'].extend(reck_20)
            data['rho'].extend([rhok for _ in reck_20])
            data['k'].extend([k for _ in reck_20])

        df = pd.DataFrame(data)
        fig = hist_plots(df, nk)
        fig.savefig('../figs/recovered_nodes/' + nk + '_avg_nr' + '.png')

# %% 5.3.7



# Run the simulation on the Contact Networks
R_vac = dict()
S_vac= dict()
Ρ_vac = dict()

# Loop over the contact networks
for nk in NETWORK_KEYS:

    if 'pres_' not in nk:

        A = NETWORKS[nk]
        β = 4e-4
        k = [1, 2, 3, 4, 5]
        Δt = 1e-2 * (1 / β)
        T = 1000
        degrees = [A.degree(n) for n in A.nodes()]
        d = sum(degrees) / len(degrees)
        ρ = None

        R_vac[nk] = []
        S_vac[nk] = []
        Ρ_vac[nk] = []

        μ = 100 * β
        mu = μ

        # 10 trials
        for t in range(10):

            # Vaccinated Nodes
            vaccinated_nodes = sorted(A.degree, key=lambda x: x[1], reverse=True)[:20]
            vac_nodes, _ =  zip(*vaccinated_nodes)
            V2 = [v for v in A.nodes() if v not in vac_nodes]
            
            # Select a random node
            v = random.sample(V2, 1)[0]
            V3 = [n for n in V2 if n != v]

            # logging.debug('Network: {} \t k: {} \t mu: {} \t v: {}'.format(nk, k, mu, v))

            # logging.debug('Simulation #{}'.format(t))

            r, s = SIR(A, β, μ, Δt, T, v, V3)
            R_vac[nk].append(r)
            S_vac[nk].append(s)

        ρ = (β / μ) * d  # ρ₀
        Ρ_vac[nk] = ρ
# %% 5.3.6 and 5.3.7
# There are no instances during our simulations where at minimum 20% of the nodes were infected
# under the Vaccination Simulation. Therefore there is nothing to plot...
# Is this an error? I have no idea, but it's interesting nonetheless

for nk in NETWORK_KEYS:
    if 'pres_' not in nk:
        A = NETWORKS[nk]
        num_nodes = A.number_of_nodes()

        recovered = R_vac[nk]
        rho = Ρ_vac[nk]

        data = {}
        data['recovered'] = []
        data['rho'] = []

        reck = recovered
        reck_20 = [r for r in reck if r > 0.2*num_nodes]

        rhok = rho

        data['recovered'].append(reck_20)
        data['rho'].append(rho)
        df = pd.DataFrame(data)
        # fig = hist_plots(df, nk)


        # fig.savefig('../figs/recovered_nodes/' + nk + '_avg_nr' + '.png')
# %%
