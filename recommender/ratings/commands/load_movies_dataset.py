import csv

from sqlalchemy.orm import sessionmaker

from ..database import engine
from ..models import Movie


def _load_base(path, model, model_index_mapping):
    Session = sessionmaker(bind=engine)
    sess = Session()

    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Remove header
        model_instances = []
        unique_ids = []

        for line in reader:
            try:
                id = int(line[model_index_mapping['id']['index']])
                if id not in unique_ids:  # There exists duplicate entries. Avoid these by adding ids to a list.
                    unique_ids.append(id)

                    # Using model_index_mapping create a dict of (attribute, value) pairs, where the attribute is the
                    # models attribute and the value is the parsed value from the input file.
                    params = {attribute: val['type'](line[val['index']])
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
            'type': int
        },
        'title': {
            'index': 20,
            'type': str
        },
        'summary': {
            'index': 9,
            'type': str
        }
    }
    _load_base(path, Movie, movies_index_mapping)
