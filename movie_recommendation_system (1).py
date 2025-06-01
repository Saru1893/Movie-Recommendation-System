
#Movie Recommendation System


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample data
data = {
    'movieId': [1, 2, 3, 4, 5],
    'title': [
        'Toy Story (1995)',
        'Jumanji (1995)',
        'Grumpier Old Men (1995)',
        'Waiting to Exhale (1995)',
        'Father of the Bride Part II (1995)'
    ],
    'genres': [
        'Adventure|Animation|Children|Comedy|Fantasy',
        'Adventure|Children|Fantasy',
        'Comedy|Romance',
        'Comedy|Drama|Romance',
        'Comedy'
    ]
}

movies = pd.DataFrame(data)

# Preprocess genres
movies['genres'] = movies['genres'].fillna('').str.replace('|', ' ', regex=False)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommender
def recommend_movie(title, num_recommendations=3):
    if title not in indices:
        return ["Movie not found."]

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]

    return movies['title'].iloc[movie_indices].tolist()

# Try it
print("Recommendations for 'Toy Story (1995)':")
for i, rec in enumerate(recommend_movie('Toy Story (1995)'), 1):
    print(f"{i}. {rec}")

