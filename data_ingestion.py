import pandas as pd
import numpy as np
import polars as pl
from config import SEED

def load_and_clean_data():
    songs_df = pd.read_csv('data/music_info.csv')
    interactions_df = pd.read_csv('data/user_listening_history.csv')
    
    for col in ['genre', 'artist']:
        if col in songs_df.columns:
            songs_df[col] = songs_df[col].fillna('Unknown')

    user_counts = interactions_df['user_id'].value_counts()
    interactions_df = interactions_df[interactions_df['user_id'].isin(user_counts[user_counts >= 5].index)]

    song_counts = interactions_df['track_id'].value_counts()
    interactions_df = interactions_df[interactions_df['track_id'].isin(song_counts[song_counts >= 3].index)]

    ranges = interactions_df.groupby('user_id')['playcount'].agg(np.ptp)
    interactions_df = interactions_df[interactions_df['user_id'].isin(ranges[ranges > 0].index)]

    songs_df = songs_df[songs_df['track_id'].isin(interactions_df['track_id'])]
    return songs_df, interactions_df

def create_labels_fast_negative_sampling(interactions_pdf, songs_pdf, num_negatives=4):
    print("\n=== LABELLING (Polars Multi-Threaded Engine) ===")
    
    # 1. Hand off the data to Polars
    interactions = pl.from_pandas(interactions_pdf)
    songs = pl.from_pandas(songs_pdf)

    # 2. Define Positives
    positives = interactions.filter(pl.col('playcount') >= 3).with_columns([
        pl.lit(1).cast(pl.Int64).alias('liked'),
        pl.col('playcount').cast(pl.Int64)
    ])

    # Calculate exact number of negatives needed per user
    user_targets = positives.group_by('user_id').agg(
        target_negs=(pl.col('track_id').count() * num_negatives).cast(pl.Int64)
    )

    # 3. The "Dart Throwing" Method (Vectorized Oversampling)
    unique_songs = songs.select('track_id').to_series().to_numpy()
    n_songs = len(unique_songs)
    
    users = user_targets['user_id'].to_numpy()
    targets = user_targets['target_negs'].to_numpy()
    
    # Generate 20% more random samples than we need to account for accidental hits
    oversampled_targets = (targets * 1.2).astype(int) + 1 
    
    # Instantly build arrays of random user/song pairs using NumPy C-bindings
    user_col = np.repeat(users, oversampled_targets)
    rng = np.random.default_rng(SEED)
    song_indices = rng.integers(0, n_songs, size=len(user_col))
    song_col = unique_songs[song_indices]
    
    candidates = pl.DataFrame({
        'user_id': user_col,
        'track_id': song_col
    })

    # 4. Filter and Finalize using Rust Multi-threading
    # Anti-join removes any random guess that the user has actually played
    true_negatives = candidates.join(
        interactions.select(['user_id', 'track_id']), 
        on=['user_id', 'track_id'], 
        how='anti'
    )
    
    # Slice it down to the exact target number per user
    final_negatives = true_negatives.with_columns(
        rn=pl.int_range(1, pl.len() + 1).over('user_id')
    ).join(
        user_targets, on='user_id'
    ).filter(
        pl.col('rn') <= pl.col('target_negs')
    ).select([
        'user_id', 
        'track_id', 
        pl.lit(0).cast(pl.Int64).alias('playcount'), 
        pl.lit(0).cast(pl.Int64).alias('liked')
    ])

    # 5. Combine, shuffle, and return to Pandas
    result = pl.concat([
        positives.select(['user_id', 'track_id', 'playcount', 'liked']), 
        final_negatives
    ])
    
    print(f"Generated {len(final_negatives)} negative samples instantly.")
    return result.sample(fraction=1.0, seed=SEED).to_pandas()

def random_user_split(full_data, train_frac=0.6, val_frac=0.2):
    rng = np.random.default_rng(SEED)
    t, v, te = [], [], []
    for _, group in full_data.groupby('user_id'):
        idx = rng.permutation(len(group))
        n = len(idx)
        t.append(group.iloc[idx[:int(n * train_frac)]])
        v.append(group.iloc[idx[int(n * train_frac):int(n * (train_frac + val_frac))]])
        te.append(group.iloc[idx[int(n * (train_frac + val_frac)):]])

    return pd.concat(t, ignore_index=True), pd.concat(v, ignore_index=True), pd.concat(te, ignore_index=True)