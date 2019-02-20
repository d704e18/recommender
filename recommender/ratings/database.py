from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('sqlite:///database.sqlite3', echo=True)  # TODO: Use remote database

Base = declarative_base()


def init_db():
    # Import models here so the metadata gets registered.
    from .models import Movie, Rating  # noqa
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
