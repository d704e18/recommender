from fastai.collab import ifnone, partial, defaults, Config
from multiprocessing import Pool
import numpy as np
import pandas as pd


class KNN:
    def __init__(self, df, categories=[]):
        self.input_index = df.index
        df = df.reset_index().drop('index', axis=1)
        self.features = df
        self.categories = None
        if categories:
            self.features = df.drop(categories, axis=1)
            self.categories = df[categories]

    def filter_categories(self, index, feature_arr, indices, k):
        elemCategories = self.categories.iloc[index]
        candidateIndices = set(self.categories.index).difference([index])

        for category in self.categories:
            discriminant = elemCategories[category]
            matching = self.categories[category] == discriminant
            if matching.sum() > k:
                newIndices = candidateIndices.intersection(
                    self.categories[matching].index)

                # Avoid excluding if less than k candidates remain as a result
                if k <= len(newIndices):
                    candidateIndices = newIndices

        candidateIndices = list(candidateIndices)

        candidates = feature_arr[candidateIndices]
        indices = indices[candidateIndices]

        return candidates, indices

    def get_candidates(self, index, k):
        feature_arr = np.asarray(self.features.values)
        indices = np.arange(0, feature_arr.shape[0])
        element = feature_arr[index, :]

        if self.categories is not None:
            candidates, indices = self.filter_categories(
                index, feature_arr, indices, k)
        else:
            # All except the index
            candidates = np.delete(feature_arr, index, 0)
            # All except the index
            indices = np.delete(indices, index, 0)

        return element, candidates, indices

    def get_knn_i(self, index, n=0, k=5):
        element, neighbors, indices = self.get_candidates(index, k)

        distances = np.sqrt(np.sum((element - neighbors)**2, axis=1))

        if len(distances) < k:
            print("What the fucking nani")

        order = np.argpartition(distances, k-1, axis=0)[:k]

        return (indices[order], distances[order])

    def get_knn(self, k=5, n_jobs=None):
        n_jobs = ifnone(n_jobs, defaults.cpus)
        indices = self.features.index
        func = partial(self.get_knn_i, k=k)
        if n_jobs > 1:
            p = Pool(n_jobs)  # use all available CPUs
            results = p.map(func, indices)
            p.close()
        else:
            results = list(map(func, indices))

        input_index = self.input_index[indices]
        neighbor_arr = np.asarray([r[0] for r in results])
        distance_arr = np.asarray([r[1] for r in results])

        return input_index, neighbor_arr, distance_arr


if __name__ == "__main__":
    path = Config.data_path()/'ml-100k'
    read_csv = partial(pd.read_csv, header=None,
                       encoding="ISO-8859-1", sep="|")
    user_columns = ["user_id", "age", "gender", "occupation", "zip_code"]
    user_df = read_csv(path/'u.user', names=user_columns)
    user_df = user_df[user_df.occupation != "other"]
    user_df['age'] = user_df['age'] * 0.1  # Scale by 0.1 to reduce impact

    item_columns = ["MovieID", "Title", "Release Date", "vrd", "IMDB", "g0",
                    "Action", "Adventure", "Animation", "Children's", "Comedy",
                    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                    "Horror", "Musical", "Mystery", "Romance",
                    "Sci-Fi", "Thriller",  "War", "Western"]

    movie_df = read_csv(path/'u.item', names=item_columns)
    movie_df.drop(["vrd", "g0", "IMDB"], axis=1, inplace=True)
    movie_df.drop(['MovieID'], axis=1, inplace=True)
    movie_df = movie_df[movie_df['Title'] != 'unknown']  # Bugged entries
    movie_df['Release Date'] = movie_df['Release Date'].apply(
        lambda x: int(x[-4:])) * 0.1  # Scale by 0.1 to reduce impact


    k = 5
    n = 1

    K_u = KNN(user_df.drop(['zip_code', 'user_id'],
                           axis=1), categories=["gender", "occupation"])
    U_indices, Ru_n, Ru_d = K_u.get_knn(k=k, n_jobs=n)

    K_i = KNN(movie_df.drop(['Title'], axis=1))
    I_indices, Ri_n, Ri_d = K_i.get_knn(k=k, n_jobs=n)
