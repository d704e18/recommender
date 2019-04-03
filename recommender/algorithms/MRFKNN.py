from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from fastai.basics import Config
from functools import partial


def ifnone(a, b):
    return a if a is not None else b


class KNN:
    def __init__(self, df, categories=None, naive=False):
        self.input_index = df.index
        df = df.reset_index().drop('index', axis=1)
        self.features = df
        self.categories = None
        self.naive = naive
        if categories:
            self.features = df.drop(categories, axis=1)
            self.categories = df[categories]

    def add_element(self, element):
        """
        this function assumes that the features in element are ordered such that
        they appear in the same order as the features in self.features, followed by
        the categories in the same order as they appear in self.categories
        """

        def row_append(df, row):
            return df.append({k: v for k, v in zip(self.features.keys(), row)}, ignore_index=True)

        _, mCont = self.features.shape
        _, mCat = self.categories.shape if self.categories is not None else (0, 0)
        if len(element) != mCont + mCat:
            raise ValueError("""dimension mismatch
                                element must have as many values as the knn instance has features and categorical variables.
                                ({0} + {1})""".format(mCont, mCat))

        feature = element[:mCont]
        category = element[mCont:]

        self.features = row_append(self.features, feature)
        self.categories = row_append(self.categories, category)

        return self.features.index[-1]

    def filter_categories(self, index, feature_arr, indices, k):
        elem_categories = self.categories.iloc[index]
        candidates = self.categories.drop(self.categories.index[index])

        for category in self.categories:
            discriminant = elem_categories[category]
            matching = candidates[category] == discriminant
            if matching.sum() >= k:
                candidates = candidates[matching]

            # Avoid excluding if less than k candidates remain as a result

        indices = indices[candidates.index]
        candidates = feature_arr[candidates.index]

        return candidates, indices

    def naive_filter_categories(self, index, feature_arr, indices, k):
        elem_categories = self.categories.iloc[index]
        candidate_indices = set(self.categories.index)
        candidate_indices.remove(index)

        for category in self.categories:
            discriminant = elem_categories[category]
            matching = self.categories[category] == discriminant
            if matching.sum() > k:
                new_indices = candidate_indices.intersection(
                    self.categories[matching].index)

                # Avoid excluding if less than k candidates remain as a result
                if k <= len(new_indices):
                    candidate_indices = new_indices

        candidate_indices = list(candidate_indices)

        candidates = feature_arr[candidate_indices]
        indices = indices[candidate_indices]

        return candidates, indices

    def get_candidates(self, index, k):
        feature_arr = np.asarray(self.features.values)
        indices = np.arange(0, feature_arr.shape[0])
        element = feature_arr[index, :]

        if self.categories is not None:
            if self.naive:
                candidates, indices = self.naive_filter_categories(
                    index, feature_arr, indices, k)
            else:
                candidates, indices = self.filter_categories(
                    index, feature_arr, indices, k)
        else:
            # All except the index
            candidates = np.delete(feature_arr, index, 0)
            # All except the index
            indices = np.delete(indices, index, 0)

        return element, candidates, indices

    def order_first_k(self, distances, k):
        order = np.argpartition(distances, k - 1, axis=0)[:k]
        order2 = np.argsort(distances[order])
        return order[order2]

    def get_knn_i(self, index, n=0, k=5):
        element, neighbors, indices = self.get_candidates(index, k)
        distances = np.sqrt(np.sum((element - neighbors) ** 2, axis=1))

        if len(distances) < k:
            print("What the fucking nani")

        order = self.order_first_k(distances, k)
        return indices[order], distances[order]

    def get_knn(self, k=5, n_jobs=None):
        n_jobs = ifnone(n_jobs, cpu_count())
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

    def query(self, element, k=5):
        old_features = self.features
        old_categories = self.categories

        index = self.add_element(element)
        result = self.get_knn_i(index, k=k)

        self.features = old_features
        self.categories = old_categories

        return result


if __name__ == "__main__":
    path = Config.data_path() / 'ml-100k'
    read_csv = partial(pd.read_csv, header=None,
                       encoding="ISO-8859-1", sep="|")
    user_columns = ["user_id", "age", "gender", "occupation", "zip_code"]
    user_df = read_csv(path / 'u.user', names=user_columns)
    user_df = user_df[user_df.occupation != "other"]
    user_df['age'] = user_df['age'] * 0.1  # Scale by 0.1 to reduce impact

    item_columns = ["MovieID", "Title", "Release Date", "vrd", "IMDB", "g0",
                    "Action", "Adventure", "Animation", "Children's", "Comedy",
                    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                    "Horror", "Musical", "Mystery", "Romance",
                    "Sci-Fi", "Thriller", "War", "Western"]

    movie_df = read_csv(path / 'u.item', names=item_columns)
    movie_df.drop(["vrd", "g0", "IMDB"], axis=1, inplace=True)
    movie_df.drop(['MovieID'], axis=1, inplace=True)
    movie_df = movie_df[movie_df['Title'] != 'unknown']  # Bugged entries
    movie_df['Release Date'] = movie_df['Release Date'].apply(
        lambda x: int(x[-4:])) * 0.1  # Scale by 0.1 to reduce impact

    k = 5

    K_u = KNN(user_df.drop(['zip_code', 'user_id'],
                           axis=1), categories=["gender", "occupation"])
    U_indices, Ru_n, Ru_d = K_u.get_knn(k=k)

    K_i = KNN(movie_df.drop(['Title'], axis=1))
    I_indices, Ri_n, Ri_d = K_i.get_knn(k=k)
