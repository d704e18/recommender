import numpy as np
import collections
import math


class DivRank(object):
    """
        Implementation of the paper: DivRank: the Interplay of Prestige and Diversity in Information Networks
        Url: https://dl.acm.org/citation.cfm?id=1835931
    """

    def rank(network : np.ndarray, steps = 100, teleport_prob : int = 0.9, follow_link_prob : int = 0.25, epsilon : int = 0.001):
        # teleport_prob is lambda in paper
        # follow_link_prob is alpha in paper
        _, n_vertices = network.shape
        p0 = np.zeros(shape=(n_vertices, n_vertices))
        pt = np.zeros(shape=(n_vertices, n_vertices))

        ranks : dict = {}
        visits : dict = {}
        preference_distribution : dict = {}

        # at step 0 every node has been "visited" once
        for v in range(0, n_vertices):
            visits[v] = 1

        # preference of visiting each node
        sumColikes = 0
        for v in range(0, n_vertices):
            # summing up each row
            preference_distribution[v] = np.sum(network[v])
            # used for normalizing
            sumColikes += np.sum(network[v])
        # normalizing visit distribution
        for v in range(0, n_vertices):
            preference_distribution[v] = \
                preference_distribution[v] / sumColikes

        # transition probabilities at step 0
        for u in range(0, n_vertices):
            for v in range(0, n_vertices):
                if u == v:
                    p0[u, v] = 1 - follow_link_prob
                else:
                    p0[u, v] = follow_link_prob * network[u, v]

        # Before iterations begin, we can compute the probability
        # of being in each node.
        # 1/n_vertices is the probability of being in
        # each node before walk has begun
        for v in range(0, n_vertices):
            sum = 0
            for u in range(0, n_vertices):
                sum += p0[u, v]
            ranks[v] = sum * 1/n_vertices

        print("--- DivRank started ---")
        # iterations begin
        for step in range(0, steps):
            newRanks : dict = {}
            DT : dict = {}

            # filling up DT with equation 4
            for u in range(0, n_vertices):
                sum = 0
                for v in range(0, n_vertices):
                    sum += p0[u, v] * visits[v]

                DT[u] = sum

            # transition probabilities at step T
            for u in range (0, n_vertices):
                for v in range(0, n_vertices):
                    pt[u, v] = \
                        (1 - teleport_prob) * preference_distribution[v] + \
                        teleport_prob * ((p0[u,v] * visits[v]) / DT[u])

            # probabilities of being each node at time T
            for v in range(0, n_vertices):
                sum = 0
                for u in range(0, n_vertices):
                    sum += pt[u, v] * ranks[u]
                newRanks[v] = sum

            # "performing walks" with equation 11
            for v in range(0, n_vertices):
                sum = 0
                for u in range(0, n_vertices):
                    sum += ((p0[u, v] * visits[v]) / DT[u]) * ranks[u]

                newRanks[v] = (1 - teleport_prob) * preference_distribution[v]\
                              + teleport_prob * sum


            # normalizing ranks in order to make sure they sum to 1
            # and not 0.9999 or similar
            factor = 1.0 / math.fsum(ranks.values())
            for k in ranks:
                ranks[k] = ranks[k] * factor

            # visit a random node.
            # Node is selected by using the probabilities of being in each node.
            nodes = np.fromiter(ranks.keys(), dtype=int)
            node = np.random.choice(nodes, 1, p=np.fromiter(ranks.values(), dtype=float))[0]
            visits[node] += 1

            delta = np.linalg.norm((np.fromiter(ranks.values(), dtype=float) - np.fromiter(newRanks.values(), dtype=float)), ord=1)

            if step % 25 == 0:
                print(f'step: {step}. delta: {delta}')

            if delta <= epsilon:
                return collections.OrderedDict(newRanks)

            ranks = newRanks




