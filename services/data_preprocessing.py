import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    PolynomialFeatures
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from typing import Dict, List, Any, Tuple, Union
import uuid
from models.preprocessing import PreprocessingConfigurationDB
from scipy import stats

class PreprocessingError(Exception):
    """Custom exception for preprocessing errors"""
    pass

class PreprocessingService:
    def __init__(self, db_session):
        self.db_session = db_session
        nltk.download('punkt')
        nltk.download('stopwords')

    def handle_missing_values(self, df: pd.DataFrame, method: str, constant_value: Any = None) -> pd.DataFrame:
        try:
            if method == "drop":
                return df.dropna()
            elif method == "mean":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                categorical_cols = df.select_dtypes(exclude=[np.number]).columns
                df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
            elif method == "median":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                categorical_cols = df.select_dtypes(exclude=[np.number]).columns
                df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
            elif method == "mode":
                for col in df.columns:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
            elif method == "constant":
                df = df.fillna(constant_value)
            elif method == "interpolate":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
            return df
        except Exception as e:
            raise PreprocessingError(f"Error handling missing values: {str(e)}")

    def scale_features(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not len(numeric_cols) or method == "none":
                return df
                
            scalers = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler(),
                "robust": RobustScaler(),
                "maxabs": MaxAbsScaler()
            }
            
            if method in scalers:
                scaler = scalers[method]
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            
            return df
        except Exception as e:
            raise PreprocessingError(f"Error scaling features: {str(e)}")

    def reduce_dimensionality(self, df: pd.DataFrame, method: str, n_components: int) -> pd.DataFrame:
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not len(numeric_cols) or method == "none":
                return df
                
            if method == "pca":
                reducer = PCA(n_components=min(n_components, len(numeric_cols)))
            elif method == "lda":
                reducer = LinearDiscriminantAnalysis(n_components=min(n_components, len(numeric_cols)-1))
            elif method == "tsne":
                reducer = TSNE(n_components=min(n_components, 3))
            else:
                return df
                
            df[numeric_cols] = reducer.fit_transform(df[numeric_cols])
            return df
        except Exception as e:
            raise PreprocessingError(f"Error reducing dimensionality: {str(e)}")

    def handle_outliers(self, df: pd.DataFrame, method: str, threshold: float) -> pd.DataFrame:
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not len(numeric_cols) or method == "none":
                return df
                
            if method == "zscore":
                z_scores = np.abs(stats.zscore(df[numeric_cols]))
                df = df[(z_scores < threshold).all(axis=1)]
            elif method == "iqr":
                Q1 = df[numeric_cols].quantile(0.25)
                Q3 = df[numeric_cols].quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                         (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
            elif method == "isolation_forest":
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(df[numeric_cols])
                df = df[outliers == 1]
            elif method == "local_outlier_factor":
                lof = LocalOutlierFactor(contamination=0.1)
                outliers = lof.fit_predict(df[numeric_cols])
                df = df[outliers == 1]
            return df
        except Exception as e:
            raise PreprocessingError(f"Error handling outliers: {str(e)}")

    def apply_sampling(self, df: pd.DataFrame, method: str, ratio: float, target_column: str = None) -> pd.DataFrame:
        try:
            if method == "none" or ratio == 1.0:
                return df
                
            X = df.drop(target_column, axis=1) if target_column else df
            y = df[target_column] if target_column else None
            
            if method == "random":
                return df.sample(frac=ratio, random_state=42)
            elif method == "stratified" and target_column:
                _, X_resampled, _, y_resampled = train_test_split(
                    X, y, train_size=ratio, stratify=y, random_state=42
                )
                return pd.concat([X_resampled, y_resampled], axis=1)
            elif method == "oversample" and target_column:
                oversample = RandomOverSampler(sampling_strategy=ratio, random_state=42)
                X_resampled, y_resampled = oversample.fit_resample(X, y)
                return pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                                pd.Series(y_resampled, name=target_column)], axis=1)
            elif method == "undersample" and target_column:
                undersample = RandomUnderSampler(sampling_strategy=ratio, random_state=42)
                X_resampled, y_resampled = undersample.fit_resample(X, y)
                return pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                                pd.Series(y_resampled, name=target_column)], axis=1)
            return df
        except Exception as e:
            raise PreprocessingError(f"Error applying sampling: {str(e)}")

    def split_data(self, df: pd.DataFrame, split_method: str, custom_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            split_ratios = {
                "70-30": 0.7,
                "80-20": 0.8,
                "90-10": 0.9,
                "custom": custom_ratio
            }
            
            ratio = split_ratios.get(split_method, 0.8)
            return train_test_split(df, train_size=ratio, random_state=42)
        except Exception as e:
            raise PreprocessingError(f"Error splitting data: {str(e)}")

    def apply_validation_split(self, df: pd.DataFrame, method: str, n_splits: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        try:
            if method == "none":
                return [(df, None)]
                
            if method == "kfold":
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            elif method == "stratified":
                kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            elif method == "timeseries":
                kf = TimeSeriesSplit(n_splits=n_splits)
            else:
                return [(df, None)]
                
            splits = []
            for train_idx, val_idx in kf.split(df):
                splits.append((df.iloc[train_idx], df.iloc[val_idx]))
            return splits
        except Exception as e:
            raise PreprocessingError(f"Error applying validation split: {str(e)}")

    def engineer_features(self, df: pd.DataFrame, techniques: List[str]) -> pd.DataFrame:
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for technique in techniques:
                if technique == "polynomial_features":
                    poly = PolynomialFeatures(degree=2, include_bias=False)
                    poly_features = poly.fit_transform(df[numeric_cols])
                    feature_names = [f"poly_{i}" for i in range(poly_features.shape[1])]
                    df = pd.concat([df, pd.DataFrame(poly_features, columns=feature_names)], axis=1)
                
                elif technique == "interaction_terms":
                    for i in range(len(numeric_cols)):
                        for j in range(i+1, len(numeric_cols)):
                            col1, col2 = numeric_cols[i], numeric_cols[j]
                            df[f"{col1}_{col2}_interaction"] = df[col1] * df[col2]
            
            return df
        except Exception as e:
            raise PreprocessingError(f"Error engineering features: {str(e)}")

    def handle_time_series(self, df: pd.DataFrame, method: str, window_size: int = 3) -> pd.DataFrame:
        try:
            if method == "none":
                return df
                
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if method == "rolling_mean":
                for col in numeric_cols:
                    df[f"{col}_rolling_mean"] = df[col].rolling(window=window_size).mean()
            
            elif method == "lag_features":
                for col in numeric_cols:
                    for lag in range(1, window_size + 1):
                        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            
            elif method == "difference":
                for col in numeric_cols:
                    df[f"{col}_diff"] = df[col].diff()
            
            return df.dropna()
        except Exception as e:
            raise PreprocessingError(f"Error handling time series: {str(e)}")

    def preprocess_text(self, df: pd.DataFrame, techniques: List[str], text_columns: List[str]) -> pd.DataFrame:
        try:
            for col in text_columns:
                if not col in df.columns:
                    continue
                    
                text_series = df[col].astype(str)
                
                if "lowercase" in techniques:
                    text_series = text_series.str.lower()
                
                if "remove_punctuation" in techniques:
                    text_series = text_series.apply(
                        lambda x: x.translate(str.maketrans("", "", string.punctuation))
                    )
                
                if "tokenize" in techniques:
                    text_series = text_series.apply(word_tokenize)
                
                if "remove_stopwords" in techniques:
                    stop_words = set(stopwords.words('english'))
                    text_series = text_series.apply(
                        lambda x: ' '.join([word for word in x.split() if word not in stop_words])
                    )
                
                df[col] = text_series
            
            return df
        except Exception as e:
            raise PreprocessingError(f"Error preprocessing text: {str(e)}")


    def validate_options(self, options: Dict) -> None:
        """Validate preprocessing options before execution"""
        try:
            # Validate missing values handling
            if options.get("missing_values_handling") not in ["drop", "mean", "median", "mode", "constant", "interpolate", "", None]:
                raise ValueError("Invalid missing values handling method")

            # Validate scaling method
            if options.get("scaling_method") not in ["standard", "minmax", "robust", "maxabs", "none", "", None]:
                raise ValueError("Invalid scaling method")

            # Validate dimensionality reduction
            if options.get("dimensionality_reduction") not in ["pca", "lda", "tsne", "none", "", None]:
                raise ValueError("Invalid dimensionality reduction method")

            # Validate n_components
            if options.get("n_components") and not isinstance(options["n_components"], int):
                raise ValueError("n_components must be a positive integer")

            # Validate outlier detection
            if options.get("outlier_detection") not in ["zscore", "iqr", "isolation_forest", "local_outlier_factor", "none", "", None]:
                raise ValueError("Invalid outlier detection method")

            # Validate outlier threshold
            if options.get("outlier_threshold") and not (1 <= float(options["outlier_threshold"]) <= 5):
                raise ValueError("Outlier threshold must be between 1 and 5")

            # Validate sampling method
            if options.get("sampling_method") not in ["random", "stratified", "oversample", "undersample", "none", "", None]:
                raise ValueError("Invalid sampling method")

            # Validate sampling ratio
            if options.get("sampling_ratio") and not (0 < float(options["sampling_ratio"]) <= 1):
                raise ValueError("Sampling ratio must be between 0 and 1")

            # Validate data split
            if options.get("data_split") not in ["70-30", "80-20", "90-10", "custom", "", None]:
                raise ValueError("Invalid data split method")

            # Validate custom split ratio
            if options.get("custom_split_ratio") and not (50 <= float(options["custom_split_ratio"]) <= 95):
                raise ValueError("Custom split ratio must be between 50 and 95")

            # Validate validation method
            if options.get("validation_method") not in ["none", "kfold", "stratified", "timeseries", "", None]:
                raise ValueError("Invalid validation method")

        except ValueError as e:
            raise PreprocessingError(f"Invalid options: {str(e)}")

    def execute_preprocessing(self, data: List[Dict[str, Any]], options: Dict) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Execute preprocessing pipeline based on configuration"""
        try:
            # Validate options before processing
            self.validate_options(options)
            
            # Convert input data to DataFrame
            df = pd.DataFrame(data)
            
            # Store original shape for logging
            original_shape = df.shape
            
            # Handle missing values
            df = self.handle_missing_values(
                df, 
                options.get("missing_values_handling", "drop"),
                options.get("constant_value")
            )
            
            # Handle duplicates
            if options.get("handling_duplicates", False):
                df = df.drop_duplicates()
            
            # Text preprocessing if specified
            if options.get("text_preprocessing"):
                text_columns = [col for col in df.columns if df[col].dtype == 'object']
                df = self.preprocess_text(df, options["text_preprocessing"], text_columns)
            
            # Time series handling
            if options.get("time_series_handling", "none") != "none":
                df = self.handle_time_series(
                    df,
                    options["time_series_handling"],
                    window_size=options.get("window_size", 3)
                )
            
            # Feature engineering
            if options.get("feature_engineering"):
                df = self.engineer_features(df, options["feature_engineering"])
            
            # Handle outliers
            if options.get("outlier_detection", "none") != "none":
                df = self.handle_outliers(
                    df,
                    options["outlier_detection"],
                    options.get("outlier_threshold", 3)
                )
            
            # Scale features
            if options.get("scaling_method", "none") != "none":
                df = self.scale_features(df, options["scaling_method"])
            
            # Reduce dimensionality
            if options.get("dimensionality_reduction", "none") != "none":
                df = self.reduce_dimensionality(
                    df,
                    options["dimensionality_reduction"],
                    options.get("n_components", 2)
                )
            
            # Apply sampling
            if options.get("sampling_method", "none") != "none":
                df = self.apply_sampling(
                    df,
                    options["sampling_method"],
                    options.get("sampling_ratio", 1.0),
                    options.get("target_column")
                )
            
            # Generate preprocessing summary
            summary = {
                "original_shape": original_shape,
                "final_shape": df.shape,
                "removed_rows": original_shape[0] - df.shape[0],
                "added_features": df.shape[1] - original_shape[1]
            }
            
            # Handle data splitting if required
            if options.get("data_split"):
                train_df, test_df = self.split_data(
                    df,
                    options["data_split"],
                    options.get("custom_split_ratio", 0.8)
                )
                
                # Apply validation split if specified
                if options.get("validation_method", "none") != "none":
                    validation_splits = self.apply_validation_split(
                        train_df,
                        options["validation_method"],
                        options.get("n_splits", 5)
                    )
                    summary["validation_splits"] = len(validation_splits)
                
                summary["train_shape"] = train_df.shape
                summary["test_shape"] = test_df.shape
                return (train_df, test_df), summary
            
            return df, summary
            
        except Exception as e:
            raise PreprocessingError(f"Failed to process data: {str(e)}")
        
        
    def store_configuration(self, options: Dict, preview_stats: Dict) -> str:
        """Store preprocessing configuration in database"""
        try:

            config = PreprocessingConfigurationDB(
                options=options,
                preview_stats=preview_stats
            )   
            self.db_session.add(config)
            self.db_session.commit()
            return config.id
        except Exception as e:
            raise PreprocessingError(f"Error storing configuration: {str(e)}")
        
        
    def get_available_options(self) -> Dict[str, Any]:
        """Return dictionary of available preprocessing options and their valid values"""
        return {
            "missing_values_handling": ["drop", "mean", "median", "mode", "constant", "interpolate", ""],
            "constant_value": "Any user-defined value",
            "handling_duplicates": [True, False],
            "scaling_method": ["standard", "minmax", "robust", "maxabs", "none", ""],
            "dimensionality_reduction": ["pca", "lda", "tsne", "none", ""],
            "n_components": "Positive integer",
            "outlier_detection": ["zscore", "iqr", "isolation_forest", "local_outlier_factor", "none", ""],
            "outlier_threshold": "Numeric value between 1 and 5 (default: 3)",
            "sampling_method": ["random", "stratified", "oversample", "undersample", "none", ""],
            "sampling_ratio": "Numeric value between 0 and 1 (default: 1.0)",
            "data_split": ["70-30", "80-20", "90-10", "custom", ""],
            "custom_split_ratio": "Numeric value between 50 and 95 (default: 80)",
            "validation_method": ["none", "kfold", "stratified", "timeseries", ""],
            "feature_engineering": ["polynomial_features", "interaction_terms"],
            "time_series_handling": ["rolling_mean", "lag_features", "difference", "none", ""],
            "text_preprocessing": ["lowercase", "remove_punctuation", "tokenize", "remove_stopwords"]
        }
