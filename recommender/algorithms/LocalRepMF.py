import pandas as pd
import numpy as np

class LocalRepMF(object):
    """
    Implementation of the paper: Local Representative-Based Matrix Factorization for Cold-Start Recommendation
    Url: https://dl.acm.org/citation.cfm?id=3108148
    """

    def fit(self,
            ratings : pd.DataFrame,
            l1 : int, # number of global questions
            l2 : int, # number of local questions
            alpha = 0.05, # regularisation param
            beta = 0.05, # regularisation param
            epochs = 1000,
            boundry = 4, # decides when a user likes or dislikes (rating >= boundry)
            k_prime = 20):
        # brug lokale variable til at holde state, i.e. V, U1, U2 osv.

        # randomize V
        # generate candidate item set Icand subset af I
        # start for loop:
            # Brug algoritme 1 til at lave G
            # for hver g i G lær U2g og Tg ved at optimere eq. 9
            # optimer V ved at bruge eq. 7
        raise NotImplementedError()

    def growTree(self): # alg. 1
        raise NotImplementedError()

    def optimizeTransformationMatrices(self): # eq. 9
        # use Maxvol to select local representatives (U2g)
        # Learn Tg
        raise NotImplementedError()

    def optimizeGlobalItemRepMatrix(self): # eq. 9
        #
        raise NotImplementedError()


if __name__ == "__main__":
    data = pd.read_csv("../ml-100k/u1.base", delimiter = "\t", header=None)
    LRMF = LocalRepMF()
    LRMF.fit(data, 2, 2)

    print("DONATELLO")

