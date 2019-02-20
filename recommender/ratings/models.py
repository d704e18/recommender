from sqlalchemy import Column, ForeignKey, Integer, String

from .database import Base


class Movie(Base):
    __tablename__ = 'movies'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    summary = Column(String)
    # TODO: Add more attributes


class Rating(Base):
    __tablename__ = 'ratings'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    movie_id = Column(Integer, ForeignKey('movies.id'))
    # TODO: Add timestamp
