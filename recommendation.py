

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CSV_PATH = 'dataset.csv'

movies_df = pd.read_csv(CSV_PATH)

if 'combined_features' not in movies_df.columns:
    movies_df['combined_features'] = movies_df.apply(lambda row: ' '.join(row.astype(str)), axis=1)

count_matrix = CountVectorizer().fit_transform(movies_df['combined_features'])
cosine_sim = cosine_similarity(count_matrix)

def get_recommendations(title):
    try:
        idx = movies_df[movies_df['title'] == title].index[0]
    except IndexError:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    
    recommendations = movies_df.iloc[movie_indices]['title'].tolist()
    return recommendations

def recommend_movies(title):
    recommendations = get_recommendations(title)
    return recommendations

recommendations = recommend_movies('Pacific Rim')
print(recommendations)