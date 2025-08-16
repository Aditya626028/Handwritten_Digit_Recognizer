

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

movies = pd.DataFrame({
    'title': ['The Matrix', 'Titanic', 'Inception', 'Avengers', 'Toy Story', 'Interstellar'],
    'genres': ['Action Sci-Fi', 'Romance Drama', 'Action Sci-Fi Thriller', 
               'Action Adventure', 'Animation Family', 'Sci-Fi Adventure']
})

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend(movie_title, cosine_sim=cosine_sim):
   
    idx = movies.index[movies['title'] == movie_title][0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:6]
    
    movie_indices = [i[0] for i in sim_scores]
    
    return movies['title'].iloc[movie_indices]

print("Recommendations for 'Inception':")
print(recommend('Inception'))

