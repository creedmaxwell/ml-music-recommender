import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

def engineer_features(train_df, val_df, test_df, songs_df):
    user_features = train_df.groupby('user_id')['playcount'].agg(
        user_mean_play='mean', user_std_play='std', user_max_play='max', user_n_songs='count'
    ).reset_index()
    
    song_features = train_df.groupby('track_id')['playcount'].agg(
        song_mean_play='mean', song_n_users='count'
    ).reset_index()

    g_user_mean = train_df['playcount'].mean()
    g_song_mean = train_df['playcount'].mean()

    def merge_fn(df):
        df = df.merge(songs_df, on='track_id', how='left')
        df = df.merge(user_features, on='user_id', how='left')
        df = df.merge(song_features, on='track_id', how='left')
        
        df.fillna({
            'user_mean_play': g_user_mean, 'user_std_play': 0, 'user_max_play': g_user_mean,
            'user_n_songs': 1, 'song_mean_play': g_song_mean, 'song_n_users': 1
        }, inplace=True)
        return df

    return merge_fn(train_df), merge_fn(val_df), merge_fn(test_df)

class CircularKeyEncoder(BaseEstimator, TransformerMixin):
    KEY_MAP = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
    def fit(self, X, y=None): 
        self.is_fitted_ = True
        return self
    def transform(self, X):
        col = pd.Series(X.iloc[:, 0] if hasattr(X, 'iloc') else X[:, 0])
        nums = pd.to_numeric(col.map(self.KEY_MAP), errors='coerce').fillna(0).astype(float)
        return np.column_stack([np.sin(2 * np.pi * nums / 12), np.cos(2 * np.pi * nums / 12)])

def build_sklearn_pipeline():
    numeric_features = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
        'instrumentalness', 'liveness', 'valence', 'tempo', 'user_mean_play', 
        'user_std_play', 'user_max_play', 'user_n_songs', 'song_mean_play', 'song_n_users'
    ]
    
    return ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric_features),
        ('bin', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', FunctionTransformer(lambda X: (X == 'major').astype(float), validate=False))]), ['mode']),
        ('key', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=0)), ('circular', CircularKeyEncoder())]), ['key']),
        ('target_enc', TargetEncoder(target_type='binary', smooth="auto"), ['genre', 'artist'])
    ], remainder='drop')