import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import uuid
from typing import Dict, List, Any
from models.preprocessing import PreprocessingConfigurationDB
from scipy import stats



class PreprocessingService:
    def __init__(self, db_session):
        self.db_session = db_session

    def store_configuration(self, config: Dict) -> str:
        """Store preprocessing configuration and return config_id"""
        db_config = PreprocessingConfigurationDB(
            options=config["options"],
            preview_stats=config["preview_stats"]
        )
        self.db_session.add(db_config)
        self.db_session.commit()
        return db_config.id
    
    def save_preprocessed_data(self, data: pd.DataFrame, location: str) -> str:
        """Save preprocessed data to a file"""
        data.to_csv(location, index=False)
        return location

    def get_configuration(self, config_id: str) -> Dict:
        """Retrieve preprocessing configuration"""
        config = self.db_session.query(PreprocessingConfigurationDB).filter_by(config_id=config_id).first()
        if not config:
            raise ValueError("Configuration not found")
        return config

    def handle_missing_values(self, df: pd.DataFrame, method: str, constant_value: str = None) -> pd.DataFrame:
        if method == "remove":
            return df.dropna()
        elif method == "mean":
            return df.fillna(df.mean())
        elif method == "median":
            return df.fillna(df.median())
        elif method == "constant":
            return df.fillna(constant_value)
        return df

    def handle_outliers(self, df: pd.DataFrame, method: str, threshold: float) -> pd.DataFrame:
        if method == "zscore":
            z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
            df = df[(z_scores < threshold).all(axis=1)]
        elif method == "iqr":
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        return df

    def scale_features(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            return df
            
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df


    def reduce_dimensionality(self, df: pd.DataFrame, method: str, n_components: int) -> pd.DataFrame:
        if method == "pca":
            pca = PCA(n_components=n_components)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = pca.fit_transform(df[numeric_cols])
        return df

    def execute_preprocessing(self, data: List[Dict[str, Any]], options: Dict) -> pd.DataFrame:
        """Execute preprocessing pipeline based on configuration"""
        df = pd.DataFrame(data)
        
        # Handle missing values
        df = self.handle_missing_values(
            df, 
            options["missing_values_handling"],
            options.get("constant_value")
        )
        
        # Handle duplicates
        if options["handling_duplicates"]:
            df = df.drop_duplicates()
        
        # Handle outliers
        if options["outlier_detection"]:
            df = self.handle_outliers(
                df,
                options["outlier_detection"],
                options["outlier_threshold"]
            )
        
        # Scale features
        if options["scaling_method"]:
            df = self.scale_features(df, options["scaling_method"])
        
        # Reduce dimensionality
        if options["dimensionality_reduction"]:
            df = self.reduce_dimensionality(
                df,
                options["dimensionality_reduction"],
                options["n_components"]
            )
        
        return df