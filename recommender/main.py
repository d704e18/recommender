from recommender.algorithms import DivRank
from recommender.algorithms.LocalRepMF import *

if __name__ == '__main__':
        data = pd.read_csv("./ml-100k/u1.base", delimiter="\t", header=None, nrows=175)  # first 2 users
        ratingsMatrix = to_matrix(data)
        tmp = colike_itemitem_network(ratingsMatrix, 4)
        dr = DivRank.DivRank()
        tmp = dr.rank(tmp)
        LRMF = LocalRepMF()

        print("DONATELLO")
