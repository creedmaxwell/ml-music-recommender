import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
import numpy as np
from sklearn.metrics import roc_auc_score
import os

class InteractionDataset(Dataset):
    """Custom Dataset for lazy loading, preventing OOM errors."""
    def __init__(self, user_ids, item_ids, labels=None):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.labels = torch.tensor(labels.values, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.user_ids[idx], self.item_ids[idx], self.labels[idx]
        return self.user_ids[idx], self.item_ids[idx]

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32, dropout=0.3):
        super(NCF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1)
        )

    def forward(self, u, i):
        return self.mlp(torch.cat([self.user_emb(u), self.item_emb(i)], dim=1)).squeeze()

def prepare_ncf_mappings(train_df, val_df, test_df):
    unique_users = train_df['user_id'].unique()
    unique_songs = train_df['track_id'].unique()
    u_map = {uid: i + 1 for i, uid in enumerate(unique_users)}
    s_map = {sid: i + 1 for i, sid in enumerate(unique_songs)}

    def map_df(df):
        u = df['user_id'].map(u_map).fillna(0).astype(int).values
        i = df['track_id'].map(s_map).fillna(0).astype(int).values
        return u, i

    u_tr, i_tr = map_df(train_df)
    u_vl, i_vl = map_df(val_df)
    u_te, i_te = map_df(test_df)
    
    return (u_tr, i_tr), (u_vl, i_vl), (u_te, i_te), len(unique_users)+1, len(unique_songs)+1

def train_ncf(train_data, val_data, y_train, y_val, num_users, num_songs, n_trials=15):
    print(f"\n=== TUNING & TRAINING PYTORCH NCF ({n_trials} trials) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()]).to(device)

    def objective(trial):
        embed_dim = trial.suggest_categorical('embed_dim', [16, 32, 64])
        lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [1024, 2048, 4096])

        model = NCF(num_users, num_songs, embed_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        tr_loader = DataLoader(InteractionDataset(train_data[0], train_data[1], y_train), batch_size=batch_size, shuffle=True)
        vl_loader = DataLoader(InteractionDataset(val_data[0], val_data[1], y_val), batch_size=batch_size)

        best_auc = 0
        for epoch in range(5): # Quick epochs for tuning
            model.train()
            for u, i, y in tr_loader:
                optimizer.zero_grad()
                loss = criterion(model(u.to(device), i.to(device)), y.to(device))
                loss.backward()
                optimizer.step()

            model.eval()
            preds = []
            with torch.no_grad():
                for u, i, _ in vl_loader:
                    preds.extend(torch.sigmoid(model(u.to(device), i.to(device))).cpu().numpy())
            
            auc = roc_auc_score(y_val, preds)
            if auc > best_auc: best_auc = auc
        return best_auc

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Final Training with Checkpointing
    best = study.best_params
    final_model = NCF(num_users, num_songs, best['embed_dim']).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=best['lr'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    tr_loader = DataLoader(InteractionDataset(train_data[0], train_data[1], y_train), batch_size=best['batch_size'], shuffle=True)
    vl_loader = DataLoader(InteractionDataset(val_data[0], val_data[1], y_val), batch_size=best['batch_size'])

    best_auc, patience_counter = 0, 0
    checkpoint_path = 'best_ncf_model.pt'

    for epoch in range(100):
        final_model.train()
        for u, i, y in tr_loader:
            optimizer.zero_grad()
            loss = criterion(final_model(u.to(device), i.to(device)), y.to(device))
            loss.backward()
            optimizer.step()

        final_model.eval()
        preds = []
        with torch.no_grad():
            for u, i, _ in vl_loader:
                preds.extend(torch.sigmoid(final_model(u.to(device), i.to(device))).cpu().numpy())
        
        auc = roc_auc_score(y_val, preds)
        if auc > best_auc:
            best_auc = auc
            torch.save(final_model.state_dict(), checkpoint_path) # Save to disk
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10: break

    final_model.load_state_dict(torch.load(checkpoint_path))
    
    # Return prediction wrapper
    def predict_proba(u_te, i_te):
        final_model.eval()
        te_loader = DataLoader(InteractionDataset(u_te, i_te), batch_size=best['batch_size'])
        preds = []
        with torch.no_grad():
            for u, i in te_loader:
                preds.extend(torch.sigmoid(final_model(u.to(device), i.to(device))).cpu().numpy())
        return np.array(preds)
        
    return predict_proba