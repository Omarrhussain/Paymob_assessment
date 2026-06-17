import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report, 
    confusion_matrix, roc_curve, precision_recall_curve
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class CampaignModelTrainer:
    def __init__(self, output_dir: str = "artifacts_model"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.campaign_cutoff = pd.to_datetime("2025-04-15", utc=True)

    def engineer_features(self, tx_df: pd.DataFrame, camp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers leakage-free features from transactions data prior to the campaign date (2025-04-15).
        Combines these features with campaign targets.
        """
        logger.info("Starting leakage-free feature engineering...")
        
        # Ensure UTC timezone awareness
        if tx_df['timestamp_utc'].dt.tz is None:
            tx_df['timestamp_utc'] = tx_df['timestamp_utc'].dt.tz_localize('UTC')
            
        # 1. Filter transactions to strictly before campaign date
        pre_camp_tx = tx_df[tx_df['timestamp_utc'] < self.campaign_cutoff].copy()
        logger.info(f"Total transactions before campaign cutoff: {len(pre_camp_tx)}")

        # Define 90d and 180d boundaries relative to the campaign date (2025-04-15)
        boundary_90d = self.campaign_cutoff - pd.Timedelta(days=90)
        boundary_180d = self.campaign_cutoff - pd.Timedelta(days=180)

        # We will build features for each customer in the campaign
        target_customers = camp_df['customer_id'].unique()
        logger.info(f"Building features for {len(target_customers)} target customers...")

        # Pre-filter transactions to targeted customers for efficiency
        pre_camp_tx = pre_camp_tx[pre_camp_tx['customer_id'].isin(target_customers)]

        # Group by customer to calculate aggregates
        features_list = []
        
        for customer_id in target_customers:
            cust_tx = pre_camp_tx[pre_camp_tx['customer_id'] == customer_id]
            
            # Default features if no transaction before campaign
            features = {
                'customer_id': customer_id,
                'recency_days': 365.0,  # Max recency default
                'frequency_90d': 0,
                'frequency_180d': 0,
                'total_spend': 0.0,
                'avg_ticket': 0.0,
                'spend_90d': 0.0,
                'active_months': 0,
                'top_mcc': 'Unknown',
                'payment_method_diversity': 0,
                'online_ratio': 0.0,
                'failure_rate': 0.0,
                'city': 'Unknown'
            }

            if not cust_tx.empty:
                # 1. Failure rate (based on all attempts: success, failed, refunded)
                total_attempts = len(cust_tx)
                failed_attempts = len(cust_tx[cust_tx['status'] == 'failed'])
                features['failure_rate'] = failed_attempts / total_attempts if total_attempts > 0 else 0.0

                # 2. Get city (take the mode of city, default to Unknown)
                cities = cust_tx['city'].dropna()
                if not cities.empty:
                    features['city'] = cities.mode().iloc[0]

                # 3. Successful transactions aggregates
                success_tx = cust_tx[cust_tx['status'] == 'success']
                if not success_tx.empty:
                    # Recency
                    last_success = success_tx['timestamp_utc'].max()
                    features['recency_days'] = float((self.campaign_cutoff - last_success).days)
                    
                    # Frequency & Monetary aggregates
                    features['total_spend'] = float(success_tx['amount_egp'].sum())
                    features['avg_ticket'] = float(success_tx['amount_egp'].mean())
                    
                    # 90d aggregates
                    success_90d = success_tx[success_tx['timestamp_utc'] >= boundary_90d]
                    features['frequency_90d'] = len(success_90d)
                    features['spend_90d'] = float(success_90d['amount_egp'].sum())

                    # 180d aggregates
                    success_180d = success_tx[success_tx['timestamp_utc'] >= boundary_180d]
                    features['frequency_180d'] = len(success_180d)

                    # Active months
                    # Get unique year_month strings
                    success_tx = success_tx.copy()
                    success_tx['year_month'] = success_tx['timestamp_utc'].dt.to_period('M').astype(str)
                    features['active_months'] = success_tx['year_month'].nunique()

                    # Top MCC
                    mccs = success_tx['mcc_category'].dropna()
                    if not mccs.empty:
                        features['top_mcc'] = mccs.mode().iloc[0]

                    # Payment Method Diversity
                    features['payment_method_diversity'] = success_tx['payment_method'].nunique()

                    # Online Ratio
                    online_count = len(success_tx[success_tx['channel'] == 'online'])
                    features['online_ratio'] = online_count / len(success_tx)
                
            features_list.append(features)

        features_df = pd.DataFrame(features_list)
        
        # Join with target from campaign responses
        features_df = features_df.merge(camp_df[['customer_id', 'responded']], on='customer_id', how='left')
        logger.info(f"Feature engineering completed. Features shape: {features_df.shape}")
        
        return features_df

    def train_and_evaluate(self, features_df: pd.DataFrame):
        """
        Trains baseline Logistic Regression and XGBoost/Random Forest models.
        Includes scaling, encoding, class imbalance handling, CV, and evaluations.
        """
        logger.info("Splitting dataset into train and test sets (80/20)...")
        
        # Target and Features separation
        X = features_df.drop(columns=['customer_id', 'responded'])
        y = features_df['responded']

        # Stratified 80/20 train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )

        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Train response rate: {y_train.mean():.2%}, Test response rate: {y_test.mean():.2%}")

        # Define numerical and categorical columns
        num_cols = [
            'recency_days', 'frequency_90d', 'frequency_180d', 'total_spend',
            'avg_ticket', 'spend_90d', 'active_months', 'payment_method_diversity',
            'online_ratio', 'failure_rate'
        ]
        cat_cols = ['top_mcc', 'city']

        # Preprocessing pipeline
        # Numerical: Scaler
        # Categorical: OneHotEncoder (handling unseen categories)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ]
        )

        # Baseline: Logistic Regression
        lr_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ])

        # Main: Random Forest Classifier
        rf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
        ])

        # Hyperparameter grids
        lr_param_grid = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0]
        }
        
        rf_param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [5, 8, 12],
            'classifier__min_samples_split': [2, 5]
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 1. Train & Tune Logistic Regression
        logger.info("Tuning Logistic Regression baseline...")
        lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        lr_grid.fit(X_train, y_train)
        best_lr = lr_grid.best_estimator_
        logger.info(f"Best LR params: {lr_grid.best_params_}, Train CV ROC-AUC: {lr_grid.best_score_:.4f}")

        # 2. Train & Tune Random Forest
        logger.info("Tuning Random Forest classifier...")
        rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        best_rf = rf_grid.best_estimator_
        logger.info(f"Best RF params: {rf_grid.best_params_}, Train CV ROC-AUC: {rf_grid.best_score_:.4f}")

        # Evaluate on test set
        self.evaluate_model(best_lr, X_test, y_test, "Logistic Regression Baseline")
        self.evaluate_model(best_rf, X_test, y_test, "Random Forest Classifier")

        # Plot ROC and Precision-Recall Curves
        self.plot_curves(best_lr, best_rf, X_test, y_test)
        
        # Plot Feature Importance for Random Forest
        self.plot_feature_importance(best_rf, num_cols, cat_cols)

        return best_rf

    def evaluate_model(self, model, X_test, y_test, model_name: str):
        """Prints classification metrics and confusion matrix."""
        logger.info(f"=== Evaluation for {model_name} ===")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"ROC-AUC: {auc:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("=" * 40)

    def plot_curves(self, lr_model, rf_model, X_test, y_test):
        """Plots and saves ROC and PR Curves for comparison."""
        plt.figure(figsize=(12, 5))

        # ROC Curve
        plt.subplot(1, 2, 1)
        for model, label in zip([lr_model, rf_model], ["Logistic Regression", "Random Forest"]):
            probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc = roc_auc_score(y_test, probs)
            plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        # Precision-Recall Curve
        plt.subplot(1, 2, 2)
        for model, label in zip([lr_model, rf_model], ["Logistic Regression", "Random Forest"]):
            probs = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, probs)
            plt.plot(recall, precision, label=label)
            
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()

        plt.tight_layout()
        plot_path = self.output_dir / "model_evaluation_curves.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved evaluation curves plot to {plot_path}")

    def plot_feature_importance(self, rf_model, num_cols, cat_cols):
        """Extracts and plots feature importances from the Random Forest model."""
        # Retrieve the transformer step and get feature names
        transformer = rf_model.named_steps['preprocessor']
        ohe = transformer.named_transformers_['cat']
        
        # Get cat column names after one-hot encoding
        ohe_cols = list(ohe.get_feature_names_out(cat_cols))
        all_features = num_cols + ohe_cols

        # Get feature importances
        importances = rf_model.named_steps['classifier'].feature_importances_
        
        feat_imp_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Plot top 15 features
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_imp_df.head(15), x='Importance', y='Feature', palette='viridis')
        plt.title("Random Forest Top 15 Feature Importances")
        plt.xlabel("Relative Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        
        plot_path = self.output_dir / "feature_importance.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved feature importance plot to {plot_path}")

if __name__ == "__main__":
    from src.data_processor import DataProcessor
    
    dp = DataProcessor()
    tx_df, camp_df = dp.process_all()
    
    trainer = CampaignModelTrainer()
    features = trainer.engineer_features(tx_df, camp_df)
    
    # Run training
    trainer.train_and_evaluate(features)
