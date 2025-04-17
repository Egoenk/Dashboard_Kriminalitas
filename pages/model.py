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
import warnings
from joblib import parallel_backend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

class CrimeRatePredictor:
    def __init__(self, test_size=0.2, n_splits=5, random_state=42, n_jobs=None):
        """
        Initialize the CrimeRatePredictor with configuration options.
        
        Parameters:
        - test_size: float, proportion of dataset to include in test split
        - n_splits: int, number of splits for time series cross-validation
        - random_state: int, random seed for reproducibility
        - n_jobs: int or None, number of jobs to run in parallel (None means 1, -1 means all)
        """
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scaler = RobustScaler()
        self.model = None
        self.feature_importances_ = None
        self.outliers = None
        self.feature_names = None

    def preprocess_data(self, df):
        """Advanced preprocessing with feature engineering and outlier handling"""
        try:
            logger.info("Starting data preprocessing")
            
            # Convert column names
            df = df.rename(columns=lambda x: x.replace(" ", "_"))
            
            # Feature Engineering
            # 1. Interaction Features
            if all(col in df.columns for col in ['educationAPK_SD', 'educationAPM_SD', 
                                               'educationAPK_SMP', 'educationAPM_SMP',
                                               'educationAPK_SMA', 'educationAPM_SMA']):
                df['education_interaction'] = (
                    df['educationAPK_SD'] * df['educationAPM_SD'] + 
                    df['educationAPK_SMP'] * df['educationAPM_SMP'] + 
                    df['educationAPK_SMA'] * df['educationAPM_SMA']
                )
            
            # 2. Polynomial Features for key predictors
            key_predictors = ['IPM_SCORE', 'kepadatan_penduduk', 'unemploymentRate']
            for feature in key_predictors:
                if feature in df.columns:
                    df[f'{feature}_squared'] = df[feature] ** 2
                    df[f'{feature}_cubed'] = df[feature] ** 3
            
            # 3. Logarithmic Transformations
            log_features = ['kepadatan_penduduk', 'unemploymentRate']
            for feature in log_features:
                if feature in df.columns:
                    # Add small constant to avoid log(0)
                    df[f'log_{feature}'] = np.log1p(df[feature])
            
            # Remove outliers using IQR method
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outliers_mask = pd.Series(False, index=df.index)
            
            for col in numeric_cols:
                if col in ['id', 'Reported_crimeRate']: 
                    continue
                
                # Use interquartile range (IQR) method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identify outliers
                outliers_mask |= (df[col] < lower_bound) | (df[col] > upper_bound)
            
            self.outliers = df[outliers_mask]
            logger.info(f"Identified {len(self.outliers)} outliers for removal")
            
            # Return cleaned dataframe
            cleaned_df = df[~outliers_mask]
            logger.info(f"Data preprocessing complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
            return cleaned_df
                
        except Exception as e:
            logger.error(f"Advanced Preprocessing error: {str(e)}", exc_info=True)
            raise

    def train_model(self, X_train, y_train):
        """Train model with feature names preservation and parallel processing safety"""
        try:
            logger.info("Starting model training")
            
            model = RandomForestRegressor(random_state=self.random_state)
            
            param_grid = {
                'n_estimators': [200],
                'max_depth': [None],
                'min_samples_split': [5],
                'min_samples_leaf': [2]
            }
            
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            
            # Configure parallel processing with safety
            with parallel_backend('loky', inner_max_num_threads=1):
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=self.n_jobs,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.feature_importances_ = pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
            
            logger.info(f"Model training complete. Best params: {grid_search.best_params_}")
            return self.model
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}", exc_info=True)
            # Fallback to sequential processing if parallel fails
            try:
                logger.warning("Parallel training failed, attempting sequential training")
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=1,  # Force sequential
                    verbose=1
                )
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                return self.model
            except Exception as fallback_e:
                logger.error(f"Fallback training also failed: {str(fallback_e)}", exc_info=True)
                raise

    def evaluate_model(self, X_test, y_test):
        """Calculate all metrics with error handling"""
        try:
            logger.info("Evaluating model")
            y_pred = self.model.predict(X_test)
            metrics = {
                'r2': r2_score(y_test, y_pred),
                'mape': mean_absolute_percentage_error(y_test, y_pred) * 100,
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            }
            logger.info(f"Model evaluation complete. Metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}", exc_info=True)
            raise

    def run_pipeline(self, df):
        """Enhanced prediction pipeline with error handling and fallbacks"""
        try:
            logger.info("Starting prediction pipeline")
            
            # 1. Preprocess data with advanced feature engineering
            df_clean = self.preprocess_data(df)
            
            # 2. Prepare features with correlation-based feature selection
            X = df_clean.drop(columns=['id', 'Reported_crimeRate'])
            y = df_clean['Reported_crimeRate'].values
            
            # Store feature names before any potential dropping
            self.feature_names = X.columns.tolist()
            
            # Correlation-based feature selection
            try:
                correlation_matrix = X.corr().abs()
                upper_tri = correlation_matrix.where(
                    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
                )
                to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
                
                if to_drop:
                    logger.info(f"Dropping highly correlated features: {to_drop}")
                    X = X.drop(columns=to_drop)
                    self.feature_names = X.columns.tolist()
            except Exception as corr_e:
                logger.warning(f"Feature correlation analysis failed: {str(corr_e)}")
            
            # 3. Split data
            split_idx = int(len(X) * (1 - self.test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # 4. Scale features
            try:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            except Exception as scale_e:
                logger.error(f"Feature scaling failed: {str(scale_e)}")
                # Fallback to unscaled data
                X_train_scaled = X_train.values
                X_test_scaled = X_test.values
            
            # 5. Train model with parallel processing safety
            self.train_model(X_train_scaled, y_train)
            
            # 6. Evaluate
            metrics = self.evaluate_model(X_test_scaled, y_test)
            
            # 7. Predict next year
            try:
                last_data = X.iloc[-1:].values
                last_data_scaled = self.scaler.transform(last_data)
                y_next, conf_interval = self.predict_future(last_data_scaled)
                
                result = {
                    'next_year_prediction': y_next[0],
                    'confidence_interval': conf_interval,
                    'metrics': metrics,
                    'feature_importances': self.feature_importances_
                }
                
                logger.info("Prediction pipeline completed successfully")
                return result
                
            except Exception as pred_e:
                logger.error(f"Prediction failed: {str(pred_e)}")
                return {
                    'error': str(pred_e),
                    'metrics': metrics,
                    'feature_importances': self.feature_importances_
                }
            
        except Exception as pipeline_e:
            logger.error(f"Pipeline failed: {str(pipeline_e)}", exc_info=True)
            return {
                'error': str(pipeline_e),
                'status': 'pipeline_failed'
            }

    def predict_future(self, X):
        """Generate prediction intervals with error handling"""
        try:
            logger.info("Generating future predictions")
            preds = self.model.predict(X)
            
            # Get predictions from all trees for confidence interval
            try:
                with parallel_backend('loky', inner_max_num_threads=1):
                    all_preds = np.stack([tree.predict(X) for tree in self.model.estimators_])
                lower = np.percentile(all_preds, 2.5, axis=0)
                upper = np.percentile(all_preds, 97.5, axis=0)
            except Exception as parallel_e:
                logger.warning(f"Parallel prediction failed, using sequential: {str(parallel_e)}")
                all_preds = np.stack([tree.predict(X) for tree in self.model.estimators_])
                lower = np.percentile(all_preds, 2.5, axis=0)
                upper = np.percentile(all_preds, 97.5, axis=0)
            
            logger.info("Future predictions generated successfully")
            return preds, (lower, upper)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise