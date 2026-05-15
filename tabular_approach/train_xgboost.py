import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

# Paths
ROOT_DIR = Path(r"d:\IIT\Academics\BCS\wheat-yield-prediction")
DATA_FILE = ROOT_DIR / "tabular_approach" / "flattened_dataset_opdate_03-05_v2.csv"

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    
    # Sort to ensure lag creation is chronologically correct per district
    df = df.sort_values(['district_id', 'season_start_year']).reset_index(drop=True)
    
    # Create lag features for past 3 years yield
    for lag in [1, 2, 3]:
        df[f'lag_{lag}_yield'] = df.groupby('district_id')['yield_kg_per_ha'].shift(lag)
        
    df['yield_delta_1'] = df['lag_1_yield'] - df['lag_2_yield']
    df['long_term_avg_yield'] = df[['lag_1_yield', 'lag_2_yield', 'lag_3_yield']].mean(axis=1)
    
    # Drop rows with NA lag_1_yield to allow for mathematical stretching
    df = df.dropna(subset=['lag_1_yield']).copy()
    
    # Define features to exclude (identifiers and leakage variables)
    exclude_cols = [
        'district_id', 'state_name', 'district_name', 'season_label', 
        'season_end_year', 'area_ha', 'production_tonnes', 
        'yield_ton_per_ha', 'yield_kg_per_ha', 'crop', 'season', 'source', 
        'source_file', 'proxy_filled', 'proxy_source_district', 'proxy_overlap_years', 
        'proxy_selection_rmse', 'proxy_selection_mae'
    ]
    
    features = [c for c in df.columns if c not in exclude_cols]
    target = 'yield_kg_per_ha'
    
    # Apply user-defined splits
    train_df = df[df['season_start_year'].between(2010, 2019)].copy()
    test_df = df[df['season_start_year'] == 2020].copy()
    val_df = df[df['season_start_year'].between(2021, 2022)].copy()
    
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    print(f"Train samples (2010-2019): {len(X_train)}")
    print(f"Test samples (2020):       {len(X_test)}")
    print(f"Val samples (2021-2022):   {len(X_val)}")
    print(f"Number of features:        {len(features)}\n")
    
    # Initialize XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='rmse'
    )
    
    # Train
    print("Training XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    # Evaluation
    VARIANCE_STRETCH_FACTOR = 1.25  # Expand predicted variance to counter conservativeness
    
    def evaluate(model, split_df, split_name):
        X = split_df[features]
        preds_raw = model.predict(X)
        
        # Apply Post-Prediction Variance Expansion (Suggestion 5)
        lag_1 = split_df['lag_1_yield'].values
        
        # Calculate raw predicted percentage change
        pred_delta_raw = (preds_raw - lag_1) / lag_1
        
        # Stretch the delta mathematically
        pred_delta_stretched = pred_delta_raw * VARIANCE_STRETCH_FACTOR
        
        # Reconstruct the absolute yield
        preds_stretched = lag_1 + (lag_1 * pred_delta_stretched)
        
        y = split_df[target].values
        
        rmse = np.sqrt(mean_squared_error(y, preds_stretched))
        mae = mean_absolute_error(y, preds_stretched)
        r2 = r2_score(y, preds_stretched)
        print(f"{split_name:25s} - Stretched RMSE: {rmse:6.2f} | MAE: {mae:6.2f} | R2: {r2:6.4f}")
        
        return preds_stretched
    
    print("\n" + "="*50)
    print(f"XGBoost Model Performance (Post-Stretch: {VARIANCE_STRETCH_FACTOR}x)")
    print("="*50)
    train_df['predicted_yield_kg_per_ha'] = evaluate(model, train_df, "Train (2010-2019)")
    val_df['predicted_yield_kg_per_ha'] = evaluate(model, val_df, "Validation (2021-2022)")
    test_df['predicted_yield_kg_per_ha'] = evaluate(model, test_df, "Test (2020)")
    print("="*50)
    
    # Feature Importance
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(importances.head(15).to_string(index=False))
    
    # Save predictions for error analysis
    print("\nSaving predictions for error analysis...")
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    all_preds = pd.concat([train_df, val_df, test_df])
    
    # Select identifying columns plus the target and prediction
    out_cols = [
        'district_id', 'state_name', 'district_name', 'season_start_year', 
        'split', 'yield_kg_per_ha', 'predicted_yield_kg_per_ha'
    ]
    
    out_file = ROOT_DIR / "tabular_approach" / "xgboost_predictions_final.csv"
    all_preds[out_cols].to_csv(out_file, index=False)
    print(f"Predictions saved to {out_file}")

if __name__ == "__main__":
    main()
