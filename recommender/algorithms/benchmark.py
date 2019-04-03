print('rewrew')
import pandas as pd
from functools import partial
from pathlib import Path
from time import time
from MRFKNN import KNN


def benchmark(df, categories, n_jobs, runs):
    print(f'Running on {n_jobs if n_jobs is not None else "all"} core(s)')
    naive_knn = KNN(df, categories, naive=True)
    new_knn = KNN(df, categories, naive=False)
    k = 10
    for i in range(runs):
        naive = time()
        naive_knn.get_knn(k=k, n_jobs=n_jobs)
        naive = time() - naive
        new = time()
        new_knn.get_knn(k=k, n_jobs=n_jobs)
        new = time() - new

        print(f'naive: {naive:.1f}\n  new: {new:.1f}')
        print(f'run {i + 1} of runs completed')


if __name__ == "__main__":
    path = Path('~/data/ml-100k').expanduser()
    read_csv = partial(pd.read_csv, header=None,
                       encoding="ISO-8859-1", sep="|")
    # user_columns = ["user_id", "age", "gender", "occupation", "zip_code"]
    # user_df = read_csv(path / 'u.user', names=user_columns)
    # user_df = user_df[user_df.occupation != "other"]
    # user_df['age'] = user_df['age'] * 0.1  # Scale by 0.1 to reduce impact

    item_columns = ["MovieID", "Title", "Release Date", "vrd", "IMDB", "g0",
                    "Action", "Adventure", "Animation", "Children's", "Comedy",
                    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                    "Horror", "Musical", "Mystery", "Romance",
                    "Sci-Fi", "Thriller", "War", "Western"]

    genres = ["Action", "Adventure", "Animation", "Children's", "Comedy",
              "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
              "Horror", "Musical", "Mystery", "Romance",
              "Sci-Fi", "Thriller", "War", "Western"]

    movie_df = read_csv(path / 'u.item', names=item_columns)
    movie_df.drop(["vrd", "g0", "IMDB"], axis=1, inplace=True)
    movie_df.drop(['MovieID'], axis=1, inplace=True)
    movie_df = movie_df[movie_df['Title'] != 'unknown']  # Bugged entries
    movie_df['Release Date'] = movie_df['Release Date'].apply(
        lambda x: int(x[-4:])) * 0.1  # Scale by 0.1 to reduce impact
    movie_df = movie_df.drop(['Title'], axis=1)

    benchmark(movie_df, genres, 1, runs=5)
    benchmark(movie_df, genres, None, runs=5)


