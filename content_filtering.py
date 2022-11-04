from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

df2 = pd.read_csv("final_movies.csv")
df2 = df2[df2['soup'].notna()]

count = CountVectorizer(stop_words="english")
count_matrix = count.fit_transform(df2['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])

def get_recommendations(title, cosine):
  idx = indices[title]
  sim_score = list(enumerate(cosine[idx]))
  sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
  sim_score = sim_score[1:11]
  movie_indices = [i[0] for i in sim_score]
  return df2[['title', 'poster_link', 'release_date', 'runtime', 'vote_average', 'overview']].iloc[movie_indices].values.tolist()
