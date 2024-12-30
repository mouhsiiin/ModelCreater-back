# services/ml_training.py
from typing import Dict, Any, Tuple, List
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

    def _prepare_data(
        self, 
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        scale_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training by splitting and scaling if required."""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        print(X.columns)
        
        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(X_train.shape, X_test.shape)
        # Scale features if requested
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Save scaler for future use
            return X_train, X_test, y_train, y_test, scaler
            
        return X_train, X_test, y_train, y_test, None

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
        
        # Validate algorithm
        if algorithm in self.regression_algorithms:
            model_class = self.regression_algorithms[algorithm]
            is_regression = True
        elif algorithm in self.classification_algorithms:
            model_class = self.classification_algorithms[algorithm]
            is_regression = False
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Prepare data
        data_split = self._prepare_data(df, target_column, test_size, scale_features)
        if scale_features:
            X_train, X_test, y_train, y_test, scaler = data_split
        else:
            X_train, X_test, y_train, y_test, _ = data_split

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

        # Save model and scaler
        model_filename = f"{algorithm}_{dataset_id}_{target_column}.pkl"
        model_path = self.models_directory / model_filename
        
        model_data = {
            "model": model,
            "feature_names": list(df.drop(columns=[target_column]).columns),
            "target_column": target_column,
            "scaler": scaler if scale_features else None
        }
        
        joblib.dump(model_data, model_path)

        return {
            "model_path": str(model_path),
            "metrics": metrics,
            "feature_names": model_data["feature_names"],
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
