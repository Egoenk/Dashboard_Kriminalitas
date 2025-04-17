import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    r2_score, mean_absolute_percentage_error,
    mean_squared_error, mean_absolute_error
)
import logging
from scipy.stats import zscore

logger = logging.getLogger(__name__)

class CrimeRatePredictor:
    def __init__(self, test_size=0.2, n_splits=5, random_state=42):
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.model = None
        self.feature_importances_ = None
        self.outliers = None

    def preprocess_data(self, df):
        """Advanced preprocessing with feature engineering"""
        try:
            # Convert column names
            df = df.rename(columns=lambda x: x.replace(" ", "_"))
            
            # Feature Engineering
            # 1. Interaction Features
            df['education_interaction'] = (
                df['educationAPK_SD'] * df['educationAPM_SD'] + 
                df['educationAPK_SMP'] * df['educationAPM_SMP'] + 
                df['educationAPK_SMA'] * df['educationAPM_SMA']
            )
            
            # 2. Polynomial Features for key predictors
            key_predictors = ['IPM_SCORE', 'kepadatan_penduduk', 'unemploymentRate']
            for feature in key_predictors:
                df[f'{feature}_squared'] = df[feature] ** 2
                df[f'{feature}_cubed'] = df[feature] ** 3
            
            # 3. Logarithmic Transformations
            log_features = ['kepadatan_penduduk', 'unemploymentRate']
            for feature in log_features:
                # Add small constant to avoid log(0)
                df[f'log_{feature}'] = np.log1p(df[feature])
            
            # Remove outliers using IQR method
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outliers_mask = pd.Series(False, index=df.index)
            
            for col in numeric_cols:
                if col in ['id', 'Reported_crimeRate']: continue
                
                # Use interquartile range (IQR) method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identify outliers
                outliers_mask |= (df[col] < lower_bound) | (df[col] > upper_bound)
            
            self.outliers = df[outliers_mask]
            
            # Return cleaned dataframe
            return df[~outliers_mask]
                
        except Exception as e:
            logger.error(f"Advanced Preprocessing error: {str(e)}")
            raise

    def train_model(self, X_train, y_train, feature_names):
        """Train model with feature names preservation"""
        try:
            model = RandomForestRegressor(random_state=self.random_state)
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.feature_importances_ = pd.Series(
                self.model.feature_importances_,
                index=feature_names
            ).sort_values(ascending=False)
            
            return self.model
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise

    def evaluate_model(self, X_test, y_test):
        """Calculate all metrics"""
        try:
            y_pred = self.model.predict(X_test)
            return {
                'r2': r2_score(y_test, y_pred),
                'mape': mean_absolute_percentage_error(y_test, y_pred) * 100,
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            }
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            raise

    def run_pipeline(self, df):
        """Enhanced prediction pipeline with feature selection"""
        try:
            # 1. Preprocess data with advanced feature engineering
            df_clean = self.preprocess_data(df)
            
            # 2. Prepare features with correlation-based feature selection
            X = df_clean.drop(columns=['id', 'Reported_crimeRate'])
            y = df_clean['Reported_crimeRate'].values
            
            # Correlation-based feature selection
            correlation_matrix = X.corr().abs()
            upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
            
            print("Highly correlated features to drop:", to_drop)
            
            X = X.drop(columns=to_drop) if to_drop else X
            feature_names = X.columns.tolist()
            
            # 3. Split data
            split_idx = int(len(X) * (1 - self.test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # 4. Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 5. Train model
            self.train_model(X_train_scaled, y_train, feature_names)
            
            # 6. Evaluate
            metrics = self.evaluate_model(X_test_scaled, y_test)
            
            # 7. Predict next year
            last_data = X.iloc[-1:].values
            last_data_scaled = self.scaler.transform(last_data)
            y_next, conf_interval = self.predict_future(last_data_scaled)
            
            return {
                'next_year_prediction': y_next[0],
                'confidence_interval': conf_interval,
                'metrics': metrics,
                'feature_importances': self.feature_importances_
            }
            
        except Exception as e:
            logger.error(f"Enhanced Pipeline error: {str(e)}")
            return None

    def predict_future(self, X):
        """Generate prediction intervals"""
        try:
            preds = self.model.predict(X)
            all_preds = np.stack([tree.predict(X) for tree in self.model.estimators_])
            lower = np.percentile(all_preds, 2.5, axis=0)
            upper = np.percentile(all_preds, 97.5, axis=0)
            return preds, (lower, upper)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise