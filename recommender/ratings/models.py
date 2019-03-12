from sqlalchemy import (BigInteger, Boolean, Column, Date, DateTime, Float,
                        ForeignKey, Integer, String, Table)
from sqlalchemy.orm import relationship

from .database import Base

movie_has_genre_table = Table('movie_has_genre', Base.metadata,
                              Column('movie_id', Integer, ForeignKey('movies.id')),
                              Column('genre_id', Integer, ForeignKey('genres.id')))

movie_has_production_company = Table('movie_has_production_company', Base.metadata,
                                     Column('movie_id', Integer, ForeignKey('movies.id')),
                                     Column('production_company_id', Integer, ForeignKey('production_companies.id')))

movie_has_production_country = Table('movie_has_production_country', Base.metadata,
                                     Column('movie_id', Integer, ForeignKey('movies.id')),
                                     Column('production_country_id', Integer, ForeignKey('production_countries.id')))

movie_has_spoken_language = Table('movie_has_spoken_language', Base.metadata,
                                  Column('movie_id', Integer, ForeignKey('movies.id')),
                                  Column('spoken_language_id', Integer, ForeignKey('spoken_languages.id')))


class Movie(Base):
    __tablename__ = 'movies'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    summary = Column(String)
    budget = Column(Integer)
    adult = Column(Boolean)
    original_language = Column(String)
    original_title = Column(String)
    poster_path = Column(String)
    release_date = Column(Date)
    revenue = Column(BigInteger)
    runtime = Column(Integer)
    status = Column(String)
    tagline = Column(String)
    video = Column(Boolean)

    ratings = relationship('Rating', back_populates='movie', cascade='all, delete-orphan')
    genres = relationship('Genre', secondary=movie_has_genre_table)
    production_companies = relationship('ProductionCompany', secondary=movie_has_production_company)
    production_countries = relationship('ProductionCountry', secondary=movie_has_production_country)
    spoken_languages = relationship('SpokenLanguage', secondary=movie_has_spoken_language)


class Rating(Base):
    __tablename__ = 'ratings'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    rating = Column(Float)
    created_at = Column(DateTime)
    movie_id = Column(Integer, ForeignKey('movies.id'))

    movie = relationship('Movie', back_populates='ratings')


class Genre(Base):
    __tablename__ = 'genres'

    id = Column(Integer, primary_key=True)
    name = Column(String)


class ProductionCompany(Base):
    __tablename__ = 'production_companies'

    id = Column(Integer, primary_key=True)
    name = Column(String)


class ProductionCountry(Base):
    __tablename__ = 'production_countries'

    id = Column(Integer, primary_key=True)
    iso_3166_1 = Column(String)
    name = Column(String)


class SpokenLanguage(Base):
    __tablename__ = 'spoken_languages'

    id = Column(Integer, primary_key=True)
    iso_639_1 = Column(String)
    name = Column(String)
