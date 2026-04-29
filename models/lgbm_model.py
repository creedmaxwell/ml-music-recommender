import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from config import SEED

def tune_and_train_lgbm(X_train, y_train, X_val, y_val, n_trials=30):
    print(f"\n=== TUNING LIGHTGBM ({n_trials} trials) ===")
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
            'random_state': SEED, 'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'scale_pos_weight': scale_pos, 'n_jobs': -1, 'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\n=== FITTING FINAL LIGHTGBM ===")
    final_params = {**study.best_params, 'n_estimators': 1000, 'objective': 'binary', 'scale_pos_weight': scale_pos, 'random_state': SEED, 'n_jobs': -1, 'verbose': -1}
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
    
    return final_model