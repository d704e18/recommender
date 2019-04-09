import numpy as np

class DivRank(object):
    """
        Implementation of the paper: DivRank: the Interplay of Prestige and Diversity in Information Networks
        Url: https://dl.acm.org/citation.cfm?id=1835931
    """

    def rank(self, network : np.ndarray, steps = 10000, teleport_prob : int = 0.9, follow_link_prob : int = 0.25, epsilon : int = 0.001):
        # teleport_prob is lambda in paper
        # follow_link_prob is alpha in paper
        _, n_vertices = network.shape
        prob_matrix = np.zeros(shape=(n_vertices, n_vertices))

        ranks : dict = {}
        visits : dict = {}

        # at step 0 every node has same rank and has been "visited" once
        for v in range(0, n_vertices):
            ranks[v] = 1 / n_vertices
            visits[v] = 1

        nodes = np.fromiter(ranks.keys(), dtype=int)

        # probabilities at step 0
        for u in range(0, n_vertices):
            for v in range(0, n_vertices):
                if u == v:
                    prob_matrix[u, v] = 1 - follow_link_prob
                else:
                    prob_matrix[u, v] = follow_link_prob * network[u, v]

        for step in range(0, steps):
            print("step: {0}".format(step))
            newRanks : dict = {}
            DT : dict = {}

            # equation 4
            for u in range(0, n_vertices):
                sum = 0
                for v in range(0, n_vertices):
                    sum += prob_matrix[u, v] * visits[v]

                DT[u] = sum

            # equation 11
            for v in range(0, n_vertices):
                sum = 0
                for u in range(0, n_vertices):
                    sum += ((prob_matrix[u, v] * visits[v]) / DT[u]) * ranks[u]

                newRanks[v] = (1 - teleport_prob) * ranks[v] + teleport_prob * sum

            # TODO: Visit a node
            node = np.random.choice(nodes, 1, p=np.fromiter(ranks.values(), dtype=float))[0]
            visits[node] += 1

            # TODO: if newRanks and ranks are similar = converged. No more steps!
            delta = np.linalg.norm((np.fromiter(ranks.values(), dtype=float) - np.fromiter(newRanks.values(), dtype=float)), ord=1)

            print(delta)
            if delta <= epsilon:
                return newRanks

            ranks = newRanks



