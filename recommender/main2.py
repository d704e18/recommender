from dataset.movielens import MovieLensDS
from algorithms.mostPopular2 import MostPopular2
from algorithms.Evaluator import Evaluator


def main2():

    print("Load data")
    generator = MovieLensDS()

    generator.items.drop(generator.items.iloc[:, 33:], inplace=True, axis=1)
    generator.items.drop(generator.items.iloc[:, 1:14], inplace=True, axis=1)
    generator.users.drop('gender_M', inplace=True, axis=1)


    mp = MostPopular2()
    algos = {"MostPopular": mp}

    print("Evaluating")
    rating_eval = Evaluator(generator)
    rating_eval.evaluate_cold_start_user(algos, 10)