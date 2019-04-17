import argparse
import os
import urllib.request
import zipfile

import config
import kaggle

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)


def load_dataset(args):
    # Ask for confirmation
    answer = input('Loading the dataset will clear the current database.\nDo you want to continue? [y/N]')
    if answer.lower() != 'y':
        print('Abort.')
        return

    # Download datasets
    ml_26m_dir = args.dir if args.dir else BASE_DIR + os.sep + 'the-movies-dataset'
    if not os.path.exists(ml_26m_dir):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('rounakbanik/the-movies-dataset', path=ml_26m_dir, unzip=True)

    ml_1m_dir = BASE_DIR + os.sep + 'ml-1m'
    if not os.path.exists(ml_1m_dir):
        urllib.request.urlretrieve('http://files.grouplens.org/datasets/movielens/ml-1m.zip', ml_1m_dir + '.zip')
        zip = zipfile.ZipFile(ml_1m_dir + '.zip', 'r')
        zip.extractall(BASE_DIR)
        zip.close()

    init_db()

    cleaned_movies_path = ml_26m_dir + os.sep + 'cleaned_movies_metadata.csv'
    preprocess_movies(ml_26m_dir + os.sep + 'movies_metadata.csv', cleaned_movies_path, ml_26m_dir + os.sep +
                      'links.csv', ml_26m_dir + os.sep + 'credits.csv')

    load_movies(cleaned_movies_path)

    # Load ratings
    if args.dataset == 'ml-26m':
        load_ratings(ml_26m_dir + os.sep + 'ratings.csv')
        mlds = MovieLensDS(l_from='db', use_ml_1m=False)
        mlds.extend_item_attributes()
        mlds.to_pkl(items_path='moviestmd.pkl', users_path='users26m.pkl', ratings_path='ratings26m.pkl')
    elif args.dataset == 'ml-1m':
        load_ratings(ml_26m_dir + os.sep + 'ratings_small.csv')
        mlds = MovieLensDS(l_from='db', use_ml_1m=True)
        mlds.extend_item_attributes()
        mlds.to_pkl(items_path='movies1m.pkl', users_path='users1m.pkl', ratings_path='ratings1m.pkl')
    elif args.dataset == 'ml-100k':
        load_ratings(ml_26m_dir + os.sep + 'ratings_small.csv')
        mlds = MovieLensDS(l_from='db', use_ml_1m=False)
        mlds.extend_item_attributes()
        mlds.to_pkl(items_path='moviestmd.pkl', users_path='users100k.pkl', ratings_path='ratings100k.pkl')


def run_algorithm(args):
    datasets = {
        'ml-1m': {
            'class': MovieLensDS,
            'params': {
                'l_from': 'pkl',
                'pkl_files': ('movies1m.pkl', 'users1m.pkl', 'ratings1m.pkl')
            }
        },
        'ml-26m': {
            'class': MovieLensDS,
            'params': {
                'l_from': 'pkl',
                'pkl_files': ('moviestmd.pkl', 'users26m.pkl', 'ratings26m.pkl')
            }
        },
        'ml-100k': {
            'class': MovieLensDS,
            'params': {
                'l_from': 'pkl',
                'pkl_files': ('moviestmd.pkl', 'users100k.pkl', 'ratings100k.pkl')
            }
        }
    }
    algorithms = {
        'mfnonvec': MFNonVec()
    }
    dataset_info = datasets.get(args.dataset)
    dataset = dataset_info['class'](**dataset_info['params'])
    algorithm = algorithms.get(args.algorithm)
    algorithm.fit(dataset.ratings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parser arguments
    parser.add_argument('--config', default='LOCAL',
                        help='Configurations to use.')
    subparsers = parser.add_subparsers(help='Sub-command help')

    # Load dataset parser
    load_ds_parser = subparsers.add_parser('load_dataset', help='Load dataset into database.')
    load_ds_parser.add_argument('--dir', nargs='?',
                                help='Directory where the dataset is located.')
    load_ds_parser.add_argument('dataset', choices=['ml-26m', 'ml-1m', 'ml-100k'])
    load_ds_parser.set_defaults(func=load_dataset)

    # Algorithm parser
    alg_parser = subparsers.add_parser('alg', help='Run recommender algorithms')
    alg_parser.add_argument('algorithm', choices=['mfnonvec'])
    alg_parser.add_argument('dataset', choices=['ml-26m', 'ml-1m', 'ml-100k'])
    alg_parser.set_defaults(func=run_algorithm)

    # Parse args and load configs
    args = parser.parse_args()
    config.set_settings(args.config)
    # Import after the configurations are set.
    from algorithms.matrix_factorization.matrix_factorization_nonVec import MatrixFactorization as MFNonVec
    from dataset.movielens import MovieLensDS
    from ratings.database import init_db
    from ratings.commands.load_movies_dataset import load_movies, load_ratings
    from ratings.commands.preprocessing import preprocess_movies

    # Call selected function with arguments
    # https://docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.add_subparsers
    args.func(args)
