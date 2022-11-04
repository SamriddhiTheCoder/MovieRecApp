import pandas as pd
import numpy as np

df = pd.read_csv("final_movies.csv")

#wr =((v/(v+m))*R)+((m/(v+m))*C)
C = df['vote_average'].mean()
m = df['vote_count'].quantile(.9)
q_movies = df.copy().loc[df['vote_count'] >= m]

def weighted_rating(x, m=m, C=C):
  v = x['vote_count']
  R = x['vote_average']
  return ((v/(v+m))*R)+((m/(v+m))*C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

output = q_movies[['title', 'poster_link', 'release_date', 'runtime', 'vote_average', 'overview']].head(20).values.tolist()
