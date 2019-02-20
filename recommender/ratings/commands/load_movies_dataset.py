import csv
from datetime import datetime

from sqlalchemy.orm import sessionmaker

from ..database import engine
from ..models import Movie, Rating


def _load_base(path, model, model_index_mapping):
    Session = sessionmaker(bind=engine)
    sess = Session()

    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Remove header
        model_instances = []
        unique_ids = []

        for line in reader:
            id = None
            try:
                if 'id' in model_index_mapping:
                    id = int(line[model_index_mapping['id']['index']])

                if id is None or id not in unique_ids:
                    # There exists duplicate entries. Avoid these by adding ids to a list.
                    unique_ids.append(id)

                    # Using model_index_mapping create a dict of (attribute, value) pairs, where the attribute is the
                    # models attribute and the value is the parsed value from the input file.
                    params = {attribute: val['parse_func'](line[val['index']])
                              for attribute, val in model_index_mapping.items()}
                    model_instances.append(model(**params))
            except (IndexError, ValueError):
                pass
        sess.add_all(model_instances)
        sess.commit()


def load_movies(path):
    # Mapping from Movie attributes to their index and type.
    movies_index_mapping = {
        'id': {
            'index': 5,
            'parse_func': int
        },
        'title': {
            'index': 20,
            'parse_func': str
        },
        'summary': {
            'index': 9,
            'parse_func': str
        }
    }
    _load_base(path, Movie, movies_index_mapping)


def load_ratings(path):
    ratings_index_mapping = {
        'user_id': {
            'index': 0,
            'parse_func': int
        },
        'movie_id': {
            'index': 1,
            'parse_func': int
        },
        'rating': {
            'index': 2,
            'parse_func': float
        },
        'created_at': {
            'index': 3,
            'parse_func': _str_to_datetime
        }
    }

    _load_base(path, Rating, ratings_index_mapping)


def _str_to_datetime(string):
    return datetime.utcfromtimestamp(int(string))
