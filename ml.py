# Import des bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt

# Charger le fichier Excel
df = pd.read_excel('donnees_evaluations_films_ml.xlsx')

# Enregistrer en tant que fichier CSV
df.to_csv('donnees_evaluations_films_ml.csv', index=False)

# Chargement des données (exemple : données sur les évaluations de films)
data = pd.read_csv('donnees_evaluations_films_ml.csv')

# Division des données en ensembles d'entraînement et de test
train_data, test_data = train_test_split(data, test_size=0.2)

# Création d'une matrice utilisateur-article (user-item matrix)
train_matrix = train_data.pivot_table(index='user_id', columns='movie_id', values='rating')

# Remplissage des valeurs manquantes avec la moyenne des évaluations de chaque utilisateur
train_matrix = train_matrix.apply(lambda x: x.fillna(x.mean()), axis=0)

# Calcul de la similarité cosine entre les utilisateurs
user_similarity = cosine_similarity(train_matrix, train_matrix)

# Fonction de prédiction basée sur la similarité entre utilisateurs
def predict(ratings, similarity):
    mean_user_rating = ratings.mean(axis=1)
    ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
    pred = mean_user_rating[:, np.newaxis] + np.dot(similarity, ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    return pred

# Prédiction des évaluations sur l'ensemble de test
test_matrix = test_data.pivot_table(index='user_id', columns='movie_id', values='rating')
user_prediction = predict(train_matrix.values, user_similarity)

# Obtenez les indices des valeurs non nulles dans le masque booléen
not_null_indices = np.where(test_matrix.notnull().values)

# Utilisez ces indices pour aplatir les matrices
user_pred_flatten = user_prediction[not_null_indices].flatten()
test_matrix_flatten = test_matrix.values[not_null_indices]

# Calcul de la racine carrée de l'erreur quadratique moyenne (RMSE)
rmse = sqrt(mean_squared_error(user_pred_flatten, test_matrix_flatten))
print('RMSE:', rmse)

print(user_prediction.shape)
print(test_matrix.shape)

# Exemple d'ajout de normalisation des évaluations par utilisateur
def normalize_ratings_by_user(ratings):
    mean_user_rating = ratings.mean(axis=1)
    normalized_ratings = ratings.sub(mean_user_rating, axis=0)
    return normalized_ratings

# Normalisation des évaluations par utilisateur dans la matrice d'entraînement
normalized_train_matrix = normalize_ratings_by_user(train_matrix)

# Recalcul de la similarité cosine avec les évaluations normalisées
normalized_user_similarity = cosine_similarity(normalized_train_matrix, normalized_train_matrix)

# Prédiction des évaluations sur l'ensemble de test avec les évaluations normalisées
normalized_user_prediction = predict(normalized_train_matrix.values, normalized_user_similarity)

# Calcul du RMSE avec les évaluations normalisées
normalized_user_pred_flatten = normalized_user_prediction[not_null_indices].flatten()
normalized_rmse = sqrt(mean_squared_error(normalized_user_pred_flatten, test_matrix_flatten))
print('Normalized RMSE:', normalized_rmse)
