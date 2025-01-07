# services/ml_training.py
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
import os
from pathlib import Path


class MLTrainingService:
    def __init__(self, models_directory: str = "trained_models/"):
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(parents=True, exist_ok=True)
        
        # Define available algorithms
        self.regression_algorithms = {
            "linear_regression": LinearRegression,
            "ridge_regression": Ridge,
            "lasso_regression": Lasso,
            "decision_tree_regressor": DecisionTreeRegressor,
            "random_forest_regressor": RandomForestRegressor,
            "svr": SVR
        }
        
        self.classification_algorithms = {
            "logistic_regression": LogisticRegression,
            "decision_tree_classifier": DecisionTreeClassifier,
            "random_forest_classifier": RandomForestClassifier,
            "svc": SVC
        }
        
        # Define columns that should be excluded from preprocessing
        self.pii_columns = {
            'id', 'name', 'email', 'phone', 'address', 'ip_address',
            'social_security', 'passport_number'
        }
        self.location_columns = {'city', 'state', 'country', 'zip_code', 'postal_code'}

    def _preprocess_categorical_features(
        self,
        df: pd.DataFrame,
        exclude_columns: List[str],
        high_cardinality_threshold: int = 50
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preprocess categorical features with intelligent handling of different column types.
        
        Args:
            df: Input DataFrame
            exclude_columns: List of columns to exclude from processing
            high_cardinality_threshold: Maximum unique values for categorical columns
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Processed DataFrame and list of dropped columns
        """
        X = df.copy()
        dropped_columns = []
        
        # Validate input columns
        existing_columns = set(X.columns)
        valid_exclude_columns = [col for col in exclude_columns if col in existing_columns]
        valid_pii_columns = self.pii_columns & existing_columns
        
        # Remove specified columns and PII
        columns_to_drop = list(set(valid_exclude_columns) | valid_pii_columns)
        print(f"Found columns in DataFrame: {existing_columns}")
        print(f"Valid columns to exclude: {valid_exclude_columns}")
        print(f"Valid PII columns found: {valid_pii_columns}")
        print(f"Final columns to drop: {columns_to_drop}")
        
        if columns_to_drop:
            X = X.drop(columns=columns_to_drop)
            dropped_columns.extend(columns_to_drop)
        
        # Handle categorical columns
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        print(f"Categorical columns found: {list(categorical_columns)}")
        
        for col in categorical_columns:
            try:
                unique_count = X[col].nunique()
                print(f"Processing column '{col}' with {unique_count} unique values")
                
                if col in self.location_columns:
                    if unique_count > high_cardinality_threshold:
                        # Frequency encoding for high-cardinality location columns
                        freq_encoding = X[col].value_counts(normalize=True)
                        X[f'{col}_freq'] = X[col].map(freq_encoding)
                        X = X.drop(columns=[col])
                        dropped_columns.append(col)
                        print(f"Applied frequency encoding to location column '{col}'")
                elif unique_count > high_cardinality_threshold:
                    # Drop high-cardinality columns
                    X = X.drop(columns=[col])
                    dropped_columns.append(col)
                    print(f"Dropped high-cardinality column '{col}'")
                else:
                    # One-hot encode low-cardinality columns
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X, dummies], axis=1)
                    X = X.drop(columns=[col])
                    print(f"One-hot encoded column '{col}'")
            except Exception as e:
                print(f"Error processing column '{col}': {str(e)}")
                continue
        
        # Verify numerical columns
        numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
        print(f"Remaining numerical columns: {list(numerical_columns)}")
        
        return X, dropped_columns

    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        scale_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler], List[str]]:
        """Prepare data for training with improved preprocessing."""
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        print("Original columns:", X.columns.tolist())
        
        # Preprocess features
        X, dropped_columns = self._preprocess_categorical_features(
            X,
            exclude_columns=[target_column]
        )
        
        print("Preprocessed columns:", X.columns.tolist())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features if requested
        scaler = None
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
        return X_train, X_test, y_train, y_test, scaler, list(X.columns)

    def _evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate regression model performance."""
        return {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2_score": float(r2_score(y_true, y_pred))
        }

    def _evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate classification model performance."""
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='weighted')),
            "recall": float(recall_score(y_true, y_pred, average='weighted')),
            "f1_score": float(f1_score(y_true, y_pred, average='weighted')),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(y_true, y_pred)
        }

    def train(
        self,
        df: pd.DataFrame,
        algorithm: str,
        target_column: str,
        dataset_id: int,
        hyperparameters: Dict[str, Any] = None,
        test_size: float = 0.2,
        scale_features: bool = True
    ) -> Dict[str, Any]:
        """Train a machine learning model with the specified algorithm and parameters."""
        
        
        print("columns:", df.columns)
        # Validate algorithm
        if algorithm in self.regression_algorithms:
            model_class = self.regression_algorithms[algorithm]
            is_regression = True
        elif algorithm in self.classification_algorithms:
            model_class = self.classification_algorithms[algorithm]
            is_regression = False
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Prepare data with improved preprocessing
        X_train, X_test, y_train, y_test, scaler, feature_names = self._prepare_data(
            df, target_column, test_size, scale_features
        )
        print("Preprocessed columns:", feature_names)

        # Initialize and train model with hyperparameters if provided
        model = model_class(**(hyperparameters or {}))
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate model
        metrics = (
            self._evaluate_regression(y_test, y_pred)
            if is_regression
            else self._evaluate_classification(y_test, y_pred)
        )

        # Save model and related data
        model_filename = f"{algorithm}_{dataset_id}_{target_column}.pkl"
        model_path = self.models_directory / model_filename
        
        model_data = {
            "model": model,
            "feature_names": feature_names,
            "target_column": target_column,
            "scaler": scaler,
            "preprocessing_info": {
                "pii_columns": list(self.pii_columns),
                "location_columns": list(self.location_columns)
            }
        }
        
        joblib.dump(model_data, model_path)

        return {
            "model_path": str(model_path),
            "metrics": metrics,
            "feature_names": feature_names,
            "is_regression": is_regression
        }


class MLPredictionService:
    def __init__(self):
        pass
        
    def _preprocess_data(self, data: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """Preprocess input data to match training features."""
        try:
            # Create a copy of input data
            df = data.copy()
            
            # Get original column names before one-hot encoding
            original_cols = df.columns
            
            # Create empty dataframe with all possible feature columns from training
            result_df = pd.DataFrame(0, index=df.index, columns=feature_names)
            
            # For each original column, handle one-hot encoding manually
            for col in original_cols:
                if col in df.columns:
                    # Get all possible encoded column names for this feature from feature_names
                    encoded_cols = [f for f in feature_names if f.startswith(f"{col}_")]
                    
                    if encoded_cols:  # If this was a categorical column in training
                        # For each row, set the appropriate one-hot encoded column to 1
                        for idx, val in df[col].items():
                            encoded_col = f"{col}_{val}"
                            if encoded_col in feature_names:
                                result_df.loc[idx, encoded_col] = 1
                    else:  # If this was a numeric column in training
                        if col in feature_names:
                            result_df[col] = df[col]
            
            # Ensure all columns are present and in correct order
            result_df = result_df[feature_names]
            
            return result_df
            
        except Exception as e:
            raise ValueError(f"Error preprocessing data: {str(e)}")
    
    def predict(self, model_path: str, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model."""
        try:
            # Load model data
            model_data = joblib.load(model_path)
            model = model_data["model"]
            scaler = model_data["scaler"]
            feature_names = model_data["feature_names"]
            
            # Print debugging information
            print("Input features:", data.columns.tolist())
            print("Expected features:", feature_names)
            
            # Preprocess input data
            X = self._preprocess_data(data, feature_names)
            
            # Verify preprocessing results
            print("Preprocessed features:", X.columns.tolist())
            
            # Scale features if necessary
            if scaler is not None:
                X = scaler.transform(X)
            
            return model.predict(X)
            
        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}\nInput columns: {data.columns.tolist()}")
