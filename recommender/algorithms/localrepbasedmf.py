import pandas as pd
import numpy as np

class LocalRepBasedMF(object):

    def __init__(self):
        return

    def loadData(self, fileName):
        data = pd.read_csv(fileName, delimiter = "\t", header=None)
        return data

if __name__ == "__main__":
    LRBMR = LocalRepBasedMF()
    ratings = LRBMR.loadData("C:\\Developer\\ml-100k\\u1.base")

