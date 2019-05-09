from recommender.algorithms import DivRank
from recommender.algorithms.LocalRepMF import *
import pickle
import os

dirname = os.path.dirname(__file__)

if __name__ == '__main__':
        path_to_data = os.path.join(dirname, "ml-100k/u1.base")
        data = pd.read_csv(path_to_data, delimiter="\t", header=None)  # first 2 users
        ratingsMatrix = to_matrix(data)

        # perform divrank on network
        path_to_ranks = os.path.join(dirname, "data/divrank_old.pkl")
        ranks_exist = os.path.isfile(path_to_ranks)
        ranks = {}
        if not ranks_exist:
            # represent network as matrix
            network = colike_itemitem_network(ratingsMatrix, 4)

            dr = DivRank.DivRank()
            ranks = dr.rank(network)
            # save to file
            file = open(path_to_ranks, "wb")
            pickle.dump(ranks, file)
            file.close
        else:
            ranks_pickle = open(path_to_ranks, "rb")
            ranks = pickle.load(ranks_pickle)

        LRMF = LocalRepMF()
        U1, U2, Ts, V = LRMF.fit(
            ratingsMatrix=ratingsMatrix,
            number_of_cand_items=200,
            l1=2,
            l2=2,
            epochs = 1000,
            boundry = 4,  # decides when a user likes or dislikes (rating >= boundry)
            k_prime = 20)

        print("DONATELLO")
