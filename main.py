import warnings
warnings.filterwarnings("ignore")

from config import seed_everything, NEGATIVE_RATIO
from data_ingestion import load_and_clean_data, create_labels_fast_negative_sampling, random_user_split
from features import engineer_features, build_sklearn_pipeline
from models.lgbm_model import tune_and_train_lgbm
from models.ncf_model import prepare_ncf_mappings, train_ncf
from evaluate import plot_individual_roc, plot_comparison_roc

def main():
    seed_everything()
    
    # 1. Data Pipeline
    songs_df, interactions_df = load_and_clean_data()
    labeled_df = create_labels_fast_negative_sampling(interactions_df, songs_df, NEGATIVE_RATIO)
    train_raw, val_raw, test_raw = random_user_split(labeled_df)
    
    y_train, y_val, y_test = train_raw['liked'], val_raw['liked'], test_raw['liked']
    
    # 2. LightGBM Pathway
    tr_b, vl_b, te_b = engineer_features(train_raw, val_raw, test_raw, songs_df)
    
    preprocessor = build_sklearn_pipeline()
    # Note: TargetEncoder fits inside the pipeline using K-Fold automatically
    X_tr_lgbm = preprocessor.fit_transform(tr_b, y_train) 
    X_vl_lgbm = preprocessor.transform(vl_b)
    X_te_lgbm = preprocessor.transform(te_b)
    
    lgbm_model = tune_and_train_lgbm(X_tr_lgbm, y_train, X_vl_lgbm, y_val, n_trials=20)
    y_proba_lgbm = lgbm_model.predict_proba(X_te_lgbm)[:, 1]
    
    # 3. PyTorch NCF Pathway
    tr_ncf, vl_ncf, te_ncf, num_users, num_songs = prepare_ncf_mappings(train_raw, val_raw, test_raw)
    ncf_predictor = train_ncf(tr_ncf, vl_ncf, y_train, y_val, num_users, num_songs, n_trials=10)
    y_proba_ncf = ncf_predictor(te_ncf[0], te_ncf[1])
    
    # 4. Evaluation
    print("\n=== GENERATING EVALUATION PLOTS ===")
    plot_individual_roc(y_test, y_proba_lgbm, "LightGBM", "lgbm_roc.png")
    plot_individual_roc(y_test, y_proba_ncf, "PyTorch NCF", "ncf_roc.png")
    plot_comparison_roc(y_test, y_proba_lgbm, y_proba_ncf, "model_comparison_roc.png")

if __name__ == "__main__":
    main()