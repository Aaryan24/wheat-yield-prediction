import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

# Paths
ROOT_DIR = Path(r"d:\IIT\Academics\BCS\wheat-yield-prediction")
DATA_FILE = ROOT_DIR / "tabular_approach" / "flattened_dataset_opdate_03-05_v2.csv"

def categorize(pct):
    if pct <= -10: return 0  # Severe Decrease
    elif pct < -5: return 1  # Significant Decrease
    elif pct <= 5: return 2  # No Significant Change
    elif pct < 10: return 3  # Significant Increase
    else: return 4           # Severe Increase

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    
    df = df.sort_values(['district_id', 'season_start_year']).reset_index(drop=True)
    
    for lag in [1, 2, 3]:
        df[f'lag_{lag}_yield'] = df.groupby('district_id')['yield_kg_per_ha'].shift(lag)
        
    df['yield_delta_1'] = df['lag_1_yield'] - df['lag_2_yield']
    df['long_term_avg_yield'] = df[['lag_1_yield', 'lag_2_yield', 'lag_3_yield']].mean(axis=1)
    
    df = df.dropna(subset=['lag_1_yield']).copy()
    
    df['actual_pct_change'] = (df['yield_kg_per_ha'] - df['lag_1_yield']) / df['lag_1_yield'] * 100
    df['target_class'] = df['actual_pct_change'].apply(categorize)
    
    exclude_cols = [
        'district_id', 'state_name', 'district_name', 'season_label', 
        'season_end_year', 'area_ha', 'production_tonnes', 
        'yield_ton_per_ha', 'yield_kg_per_ha', 'crop', 'season', 'source', 
        'source_file', 'proxy_filled', 'proxy_source_district', 'proxy_overlap_years', 
        'proxy_selection_rmse', 'proxy_selection_mae',
        'actual_pct_change', 'target_class'
    ]
    
    features = [c for c in df.columns if c not in exclude_cols]
    target_reg = 'yield_kg_per_ha'
    target_class = 'target_class'
    
    train_df = df[df['season_start_year'].between(2010, 2019)].copy()
    test_df = df[df['season_start_year'] == 2020].copy()
    val_df = df[df['season_start_year'].between(2021, 2022)].copy()
    
    X_train, y_train_reg, y_train_class = train_df[features], train_df[target_reg], train_df[target_class]
    X_val, y_val_reg, y_val_class = val_df[features], val_df[target_reg], val_df[target_class]
    X_test, y_test_reg, y_test_class = test_df[features], test_df[target_reg], test_df[target_class]
    
    print("\n--- STEP 1: Training XGBoost Regressor ---")
    regressor = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        early_stopping_rounds=50, eval_metric='rmse'
    )
    regressor.fit(X_train, y_train_reg, eval_set=[(X_val, y_val_reg)], verbose=False)
    
    train_pred_yield = regressor.predict(X_train)
    val_pred_yield = regressor.predict(X_val)
    test_pred_yield = regressor.predict(X_test)
    
    # We pass the predicted percentage change as the feature to the classifier
    # so it knows if the regressor predicted an increase or decrease.
    X_train_class_feat = pd.DataFrame({'pred_pct_change': (train_pred_yield - train_df['lag_1_yield']) / train_df['lag_1_yield'] * 100})
    X_val_class_feat = pd.DataFrame({'pred_pct_change': (val_pred_yield - val_df['lag_1_yield']) / val_df['lag_1_yield'] * 100})
    X_test_class_feat = pd.DataFrame({'pred_pct_change': (test_pred_yield - test_df['lag_1_yield']) / test_df['lag_1_yield'] * 100})
    
    print("\n--- STEP 2: Training XGBoost Classifier on Predicted % Change ---")
    classifier = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3,
        random_state=42, objective='multi:softprob', num_class=5
    )
    classifier.fit(X_train_class_feat, y_train_class)
    
    class_names = [
        'Severe Dec (<= -10%)',
        'Signif Dec (-10% to -5%)',
        'No Signif Change',
        'Signif Inc (5% to 10%)',
        'Severe Inc (>= 10%)'
    ]
    
    def evaluate(model, X, y, split_name):
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        print(f"\n{'='*50}\n{split_name} Performance (Accuracy: {acc:.4f})\n{'='*50}")
        print(classification_report(y, preds, target_names=class_names, zero_division=0))
    
    evaluate(classifier, X_train_class_feat, y_train_class, "Train (2010-2019)")
    evaluate(classifier, X_val_class_feat, y_val_class, "Validation (2021-2022)")
    evaluate(classifier, X_test_class_feat, y_test_class, "Test (2020)")

if __name__ == "__main__":
    main()
