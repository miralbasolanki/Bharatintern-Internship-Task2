import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

print(movies.columns)  # Check if 'movieId' is a column in movies
print(ratings.columns)  # Check if 'movieId' is a column in ratings

data = pd.merge(ratings, movies, on='movieId')

user_item_matrix = data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

user_ratings_mean = np.mean(user_item_matrix.values, axis=1)
user_item_matrix_demeaned = user_item_matrix.values - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(user_item_matrix_demeaned, k=50)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
predicted_ratings_df = pd.DataFrame(all_user_predicted_ratings, columns=user_item_matrix.columns)


def recommend_movies(predictions_df, user_id, movies_df, ratings_df, num_recommendations=10):

    user_row_number = user_id - 1
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    user_data = ratings_df[ratings_df.userId == user_id]
    user_full = user_data.merge(movies_df, how='left', on='movieId').sort_values(['rating'], ascending=False)

    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
                       merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                             left_on='title', right_on='title').
                       rename(columns={user_row_number: 'Predictions'}).
                       sort_values('Predictions', ascending=False).
                       iloc[:num_recommendations, :-1])

    return user_full, recommendations


already_rated, predictions = recommend_movies(predicted_ratings_df, 1, movies, ratings, 10)

print("Movies already rated by user 1:")
print(already_rated)

print("\nTop 10 movie recommendations for user 1:")
print(predictions)
