import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import pandas as pd
import numpy as np
import joblib

import matplotlib
matplotlib.use('Agg') # Thread-safe plotting
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    confusion_matrix, precision_score, recall_score,
    average_precision_score, classification_report,
    roc_curve, auc
)

# ── Ensure LightGBM is installed ──
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except ImportError:
    raise ImportError("LightGBM is required. Install with: pip install lightgbm")

# ── Ensure PyTorch is installed ──
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    raise ImportError("PyTorch is required. Install with: pip install torch")

# Silence the LightGBM feature names warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")


# ═══════════════════════════════════════════════════════════
# 1-3. DATA LOADING, CLEANING & LABELLING
# ═══════════════════════════════════════════════════════════

def load_data():
    songs_df = pd.read_csv('data/music_info.csv')
    interactions_df = pd.read_csv('data/user_listening_history.csv')
    return songs_df, interactions_df

def sample_users(interactions_df, target_interactions=100_000, random_state=42):
    avg = interactions_df.groupby('user_id').size().mean()
    target_users = int(target_interactions / avg)
    users = interactions_df['user_id'].unique()
    rng = np.random.default_rng(random_state)
    chosen = rng.choice(users, size=min(target_users, len(users)), replace=False)
    return interactions_df[interactions_df['user_id'].isin(chosen)].copy()

def clean_data(songs_df, interactions_df):
    print("=== CLEANING ===")
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
    print(f"Songs: {songs_df.shape}  |  Interactions: {interactions_df.shape}")
    return songs_df, interactions_df

def create_labels_negative_sampling(interactions_df, songs_df, num_negatives=4):
    print("\n=== LABELLING (Negative Sampling) ===")
    
    # 1. Define Positives
    # Filter out songs played only 1 or 2 times (could be skips/accidents)
    # Everything >= 3 plays is a solid positive signal.
    positives = interactions_df[interactions_df['playcount'] >= 3].copy()
    positives['liked'] = 1
    
    # 2. Define Negatives
    all_users = positives['user_id'].unique()
    all_songs = songs_df['track_id'].unique()
    
    negatives_list = []
    
    # Use a random number generator for speed
    rng = np.random.default_rng(42)
    
    for user in all_users:
        # Get songs this user has actually played
        user_played = set(interactions_df[interactions_df['user_id'] == user]['track_id'])
        
        # Determine how many negatives to sample (e.g., 1 negative for every 1 positive)
        num_user_positives = len(positives[positives['user_id'] == user])
        target_negatives = num_user_positives * num_negatives
        
        sampled_negs = 0
        user_negatives = []
        
        # Randomly sample songs until we find enough they haven't played
        while sampled_negs < target_negatives:
            candidate = rng.choice(all_songs)
            if candidate not in user_played:
                user_negatives.append({'user_id': user, 'track_id': candidate, 'playcount': 0, 'liked': 0})
                sampled_negs += 1
                
        negatives_list.extend(user_negatives)
        
    negatives = pd.DataFrame(negatives_list)
    
    # 3. Combine and shuffle
    result = pd.concat([positives, negatives], ignore_index=True)
    result = result.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle
    
    print(f"Users: {result['user_id'].nunique():,}")
    print(f"Interactions: {len(result):,}")
    print(f"Class balance: {result['liked'].mean():.2%} positive")
    
    return result

def per_user_split(full_data, train=0.6, val=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    t, v, te = [], [], []
    for uid, group in full_data.groupby('user_id'):
        idx = rng.permutation(len(group))
        n = len(idx)
        t.append(group.iloc[idx[:int(n * train)]])
        v.append(group.iloc[idx[int(n * train):int(n * (train + val))]])
        te.append(group.iloc[idx[int(n * (train + val)):]])

    print(f"\n=== SPLIT ===")
    return pd.concat(t, ignore_index=True), pd.concat(v, ignore_index=True), pd.concat(te, ignore_index=True)


# ═══════════════════════════════════════════════════════════
# 4. LIGHTGBM PATHWAY: FEATURE ENGINEERING & PREPROCESSING
# ═══════════════════════════════════════════════════════════

def calculate_and_merge_features(train_df, val_df, test_df, songs_df):
    user_features = train_df.groupby('user_id')['playcount'].agg(
        user_mean_play='mean', user_std_play='std', user_max_play='max', user_n_songs='count'
    ).reset_index()
    user_features['user_std_play'] = user_features['user_std_play'].fillna(0)

    song_features = train_df.groupby('track_id')['playcount'].agg(
        song_mean_play='mean', song_n_users='count'
    ).reset_index()

    global_user_mean = train_df['playcount'].mean()
    global_song_mean = train_df['playcount'].mean()

    def merge_features(df):
        df = df.copy()
        df = df.merge(songs_df, on='track_id', how='left')
        df = df.merge(user_features, on='user_id', how='left')
        df = df.merge(song_features, on='track_id', how='left')
        df['user_mean_play'] = df['user_mean_play'].fillna(global_user_mean)
        df['user_std_play']  = df['user_std_play'].fillna(0)
        df['user_max_play']  = df['user_max_play'].fillna(global_user_mean)
        df['user_n_songs']   = df['user_n_songs'].fillna(1)
        df['song_mean_play'] = df['song_mean_play'].fillna(global_song_mean)
        df['song_n_users']   = df['song_n_users'].fillna(1)
        return df

    return merge_features(train_df), merge_features(val_df), merge_features(test_df)

class CircularKeyEncoder(BaseEstimator, TransformerMixin):
    KEY_MAP = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
    def fit(self, X, y=None):
        self.is_fitted_ = True; return self
    def transform(self, X):
        col = pd.Series(X.iloc[:, 0] if hasattr(X, 'iloc') else X[:, 0])
        nums = col.map(self.KEY_MAP).fillna(0).astype(float) if col.dtype == object else pd.to_numeric(col, errors='coerce').fillna(0).astype(float)
        return np.column_stack([np.sin(2 * np.pi * nums / 12), np.cos(2 * np.pi * nums / 12)])

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=10): self.smoothing = smoothing
    def fit(self, X, y):
        df = pd.DataFrame({'col': X.iloc[:, 0] if hasattr(X, 'iloc') else X[:, 0], 'target': y})
        self.global_mean_ = df['target'].mean()
        stats = df.groupby('col')['target'].agg(['mean', 'count'])
        stats['encoded'] = ((stats['mean'] * stats['count'] + self.global_mean_ * self.smoothing) / (stats['count'] + self.smoothing))
        self.mapping_ = stats['encoded'].to_dict()
        return self
    def transform(self, X):
        col = pd.Series(X.iloc[:, 0] if hasattr(X, 'iloc') else X[:, 0])
        return col.map(self.mapping_).fillna(self.global_mean_).to_numpy().reshape(-1, 1)

def apply_target_encoding(X_train, y_train, X_val, X_test, cat_cols=None, smoothing_map=None):
    if not smoothing_map: smoothing_map = {'genre': 20, 'artist': 10}
    for col in cat_cols:
        if col not in X_train.columns: continue
        enc = TargetEncoder(smoothing=smoothing_map.get(col, 10))
        enc.fit(X_train[[col]].fillna('Unknown'), y_train)
        enc_col = f'{col}_enc'
        for df in [X_train, X_val, X_test]:
            df[enc_col] = enc.transform(df[[col]].fillna('Unknown')).ravel()
            df.drop(columns=[col], inplace=True)
    return X_train, X_val, X_test

def build_pipeline():
    numeric_features = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
        'user_mean_play', 'user_std_play', 'user_max_play', 'user_n_songs', 'song_mean_play', 'song_n_users', 'genre_enc', 'artist_enc',
    ]
    return ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric_features),
        ('bin', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', FunctionTransformer(lambda X: (X == 1).astype(float) if X.dtype != object else (X == 'major').astype(float), validate=False))]), ['mode']),
        ('key', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=0)), ('circular', CircularKeyEncoder())]), ['key']),
    ], remainder='drop')


def tune_lightgbm(X_train_pre, y_train, X_val_pre, y_val, n_trials=30, random_state=42):
    print(f"\n=== TUNING LIGHTGBM ({n_trials} trials) ===")
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    def objective(trial):
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
            'random_state': random_state, 'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'scale_pos_weight': neg / pos, 'n_jobs': -1, 'verbose': -1
        }
        
        model = LGBMClassifier(**params)
        model.fit(X_train_pre, y_train, eval_set=[(X_val_pre, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
        y_proba = model.predict_proba(X_val_pre)[:, 1]
        return roc_auc_score(y_val, y_proba)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best LightGBM val AUC-ROC : {study.best_value:.4f}")
    return study.best_params


# ═══════════════════════════════════════════════════════════
# 5. PYTORCH NCF PATHWAY: ID MAPPING & EMBEDDINGS
# ═══════════════════════════════════════════════════════════

def prepare_ncf_data(train_df, val_df, test_df):
    """ Maps string user_ids and track_ids to contiguous integers for embedding layers """
    unique_users = train_df['user_id'].unique()
    unique_songs = train_df['track_id'].unique()

    # Index 0 is reserved for UNKNOWN users/songs
    user_map = {uid: i + 1 for i, uid in enumerate(unique_users)}
    song_map = {sid: i + 1 for i, sid in enumerate(unique_songs)}

    def apply_mapping(df):
        u_idx = df['user_id'].map(user_map).fillna(0).astype(int).values
        s_idx = df['track_id'].map(song_map).fillna(0).astype(int).values
        return np.column_stack([u_idx, s_idx])

    X_tr = apply_mapping(train_df)
    X_vl = apply_mapping(val_df)
    X_te = apply_mapping(test_df)

    num_users = len(unique_users) + 1
    num_songs = len(unique_songs) + 1

    return X_tr, X_vl, X_te, num_users, num_songs

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32, dropout_rate=0.3):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        # The Embedding layers (Lookup Tables)
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embed_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embed_dim)
        
        # Dense layers to learn interactions between user and item embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(32, 1) # Raw logits output
        )

    def forward(self, x):
        # x is a batch of [user_idx, item_idx]
        user_idx = x[:, 0]
        item_idx = x[:, 1]
        
        # Look up the multi-dimensional vectors for these specific users and songs
        user_vector = self.user_embedding(user_idx)
        item_vector = self.item_embedding(item_idx)
        
        # Concatenate them side-by-side
        concat_vector = torch.cat([user_vector, item_vector], dim=1)
        
        # Pass through the network to get probability score
        return self.mlp(concat_vector)

class NCF_Wrapper:
    """ Wraps the NCF model so it behaves like sklearn. """
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.long).to(self.device)
            # Use chunks to prevent OOM errors on large test sets
            logits = []
            for chunk in torch.chunk(X_tensor, chunks=10, dim=0):
                chunk_logits = self.model(chunk)
                logits.append(chunk_logits)
                
            logits = torch.cat(logits, dim=0)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
        return np.column_stack([1 - probs, probs])

def tune_and_train_ncf(X_train, y_train, X_val, y_val, num_users, num_songs, n_trials=15, max_epochs=50):
    print(f"\n=== TUNING PYTORCH NCF ({n_trials} trials) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_tr_t = torch.tensor(X_train, dtype=torch.long)
    y_tr_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_vl_t = torch.tensor(X_val, dtype=torch.long)
    y_vl_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    pos_weight = torch.tensor([(len(y_tr_t) - y_tr_t.sum()) / y_tr_t.sum()]).to(device)

    def objective(trial):
        embed_dim = trial.suggest_categorical('embed_dim', [16, 32, 64])
        lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        # NCF benefits from large batch sizes
        batch_size = trial.suggest_categorical('batch_size', [1024, 2048, 4096]) 

        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_vl_t, y_vl_t), batch_size=batch_size, shuffle=False)

        model = NeuralCollaborativeFiltering(num_users, num_songs, embed_dim, dropout).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        best_auc = 0
        patience, patience_counter = 5, 0

        for epoch in range(max_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            val_preds = []
            with torch.no_grad():
                for X_batch, _ in val_loader:
                    val_preds.extend(torch.sigmoid(model(X_batch.to(device))).cpu().numpy())
            
            val_auc = roc_auc_score(y_val, val_preds)
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
                
        return best_auc

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best NCF val AUC-ROC : {study.best_value:.4f}")
    
    # --- TRAIN FINAL MODEL ---
    print("\n=== FITTING FINAL PYTORCH NCF MODEL ===")
    best = study.best_params
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=best['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_vl_t, y_vl_t), batch_size=best['batch_size'], shuffle=False)
    
    final_model = NeuralCollaborativeFiltering(num_users, num_songs, best['embed_dim'], best['dropout']).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(final_model.parameters(), lr=best['lr'], weight_decay=1e-5)
    
    best_final_auc, best_weights = 0, None
    patience, patience_counter = 10, 0

    for epoch in range(100):
        final_model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(final_model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        final_model.eval()
        val_preds = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                val_preds.extend(torch.sigmoid(final_model(X_batch.to(device))).cpu().numpy())
        
        val_auc = roc_auc_score(y_val, val_preds)
        if val_auc > best_final_auc:
            best_final_auc = val_auc
            best_weights = final_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break

    final_model.load_state_dict(best_weights)
    return NCF_Wrapper(final_model, device)


# ═══════════════════════════════════════════════════════════
# 6. EVALUATION PLOTS
# ═══════════════════════════════════════════════════════════

def plot_combined_roc_curves(lgbm_model, X_te_lgbm, ncf_model, X_te_ncf, y_test, filename='lgbm_vs_ncf_roc.png'):
    plt.figure(figsize=(8, 6))
    
    # LightGBM Curve
    y_proba_lgbm = lgbm_model.predict_proba(X_te_lgbm)[:, 1]
    fpr_l, tpr_l, _ = roc_curve(y_test, y_proba_lgbm)
    plt.plot(fpr_l, tpr_l, lw=2, label=f'LightGBM (Tabular Audio) (AUC = {auc(fpr_l, tpr_l):.3f})')
    
    # NCF Curve
    y_proba_ncf = ncf_model.predict_proba(X_te_ncf)[:, 1]
    fpr_n, tpr_n, _ = roc_curve(y_test, y_proba_ncf)
    plt.plot(fpr_n, tpr_n, lw=2, label=f'PyTorch NCF (Embeddings) (AUC = {auc(fpr_n, tpr_n):.3f})')
        
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Guessing')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold'); plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('Content-Based vs. Collaborative Filtering', fontweight='bold', pad=15)
    plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


# ═══════════════════════════════════════════════════════════
# 7. FULL PIPELINE
# ═══════════════════════════════════════════════════════════

def full_pipeline(sample_size='small'):
    targets = {'small': 100_000, 'medium': 500_000, 'large': 2_000_000, 'full': None}
    
    # 1. Prepare Core Data
    songs_df, interactions_df = load_data()
    if targets[sample_size] is not None: interactions_df = sample_users(interactions_df, targets[sample_size])
    songs_df, interactions_df = clean_data(songs_df, interactions_df)
    labeled = create_labels_negative_sampling(interactions_df, songs_df)
    train_df_raw, val_df_raw, test_df_raw = per_user_split(labeled)
    y_train, y_val, y_test = train_df_raw['liked'], val_df_raw['liked'], test_df_raw['liked']
    
    # ==========================================
    # PATHWAY A: LightGBM on Tabular Features
    # ==========================================
    print("\n--- INITIATING LIGHTGBM PATHWAY ---")
    tr_b, vl_b, te_b = calculate_and_merge_features(train_df_raw, val_df_raw, test_df_raw, songs_df)

    feature_cols_base = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'user_mean_play', 'user_std_play', 'user_max_play', 'user_n_songs', 'song_mean_play', 'song_n_users', 'mode', 'key', 'genre', 'artist']
    feature_cols_base = [c for c in feature_cols_base if c in tr_b.columns]

    X_train_lgbm, X_val_lgbm, X_test_lgbm = tr_b[feature_cols_base].copy(), vl_b[feature_cols_base].copy(), te_b[feature_cols_base].copy()

    print("Target Encoding...")
    X_train_lgbm, X_val_lgbm, X_test_lgbm = apply_target_encoding(X_train_lgbm, y_train, X_val_lgbm, X_test_lgbm, cat_cols=['genre', 'artist'])

    print("Scaling and Imputing...")
    preprocessor = build_pipeline()
    X_tr_pre_lgbm = preprocessor.fit_transform(X_train_lgbm, y_train)
    X_vl_pre_lgbm = preprocessor.transform(X_val_lgbm)
    X_te_pre_lgbm = preprocessor.transform(X_test_lgbm)

    best_lgbm_params = tune_lightgbm(X_tr_pre_lgbm, y_train, X_vl_pre_lgbm, y_val, n_trials=30)
    print("\nTraining Final LightGBM...")
    final_lgbm_params = {**best_lgbm_params, 'n_estimators': 1000, 'objective': 'binary', 'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(), 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
    lgbm_model = LGBMClassifier(**final_lgbm_params)
    lgbm_model.fit(X_tr_pre_lgbm, y_train, eval_set=[(X_vl_pre_lgbm, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])

    # ==========================================
    # PATHWAY B: PyTorch Neural Collaborative Filtering
    # ==========================================
    print("\n--- INITIATING PYTORCH NCF PATHWAY ---")
    X_tr_ncf, X_vl_ncf, X_te_ncf, num_users, num_songs = prepare_ncf_data(train_df_raw, val_df_raw, test_df_raw)
    
    ncf_wrapper = tune_and_train_ncf(X_tr_ncf, y_train, X_vl_ncf, y_val, num_users, num_songs, n_trials=15)

    # ==========================================
    # FINAL EVALUATION
    # ==========================================
    print("\n=== FINAL MODEL PERFORMANCE (TEST SET) ===")
    
    # Eval LightGBM
    y_proba_lgbm = lgbm_model.predict_proba(X_te_pre_lgbm)[:, 1]
    y_pred_lgbm  = (y_proba_lgbm >= 0.5).astype(int)
    print(f"\n--- LightGBM (Tabular Audio Features) ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred_lgbm):.4f}")
    print(f"AUC-ROC  : {roc_auc_score(y_test, y_proba_lgbm):.4f}")
    print(f"Avg Prec : {average_precision_score(y_test, y_proba_lgbm):.4f}")

    # Eval NCF
    y_proba_ncf = ncf_wrapper.predict_proba(X_te_ncf)[:, 1]
    y_pred_ncf  = (y_proba_ncf >= 0.5).astype(int)
    print(f"\n--- PyTorch NCF (Collaborative Embeddings) ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred_ncf):.4f}")
    print(f"AUC-ROC  : {roc_auc_score(y_test, y_proba_ncf):.4f}")
    print(f"Avg Prec : {average_precision_score(y_test, y_proba_ncf):.4f}")

    plot_combined_roc_curves(lgbm_model, X_te_pre_lgbm, ncf_wrapper, X_te_ncf, y_test)

if __name__ == '__main__':
    full_pipeline(sample_size='small')