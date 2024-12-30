import os
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score

from database.base import get_db
from routers.dataset import UPLOAD_DIRECTORY
from fastapi import FastAPI
from fastapi.responses import FileResponse
from io import BytesIO
from reportlab.pdfgen import canvas
from fastapi import HTTPException

router = APIRouter(prefix="/auto", tags=["auto_Crafter"])

@router.post("/craft")
async def auto_ml_pipeline(
    file: UploadFile = File(...)
):
    # Save file
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Read and validate dataset
    try:
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file_path)
        else:
            data = pd.read_json(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    try:
        # Preprocessing: Handle missing values
        imputer = SimpleImputer(strategy='mean')
        
        # Identify numeric and categorical columns
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        # Impute missing values for numeric columns
        if not numeric_columns.empty:
            data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
        
        # Encode categorical columns
        if not categorical_columns.empty:
            label_encoders = {}
            for col in categorical_columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                label_encoders[col] = le

        print("Preprocessing completed successfully!")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

    # Debugging shapes of X and y
    X = data.iloc[:, :-1]  # All columns except the last as features
    y = data.iloc[:, -1]   # Last column as the target variable

    print(f"Shape of X before split: {X.shape}")
    print(f"Shape of y before split: {y.shape}")    

    # Determine if the task is regression or classification
    if y.nunique() < 20:  # Adjust threshold for discrete values
        # Classification task
        task_type = "classification"
        print("Classification task detected.")
    else:
        # Regression task
        task_type = "regression"
        print("Regression task detected.")

    try:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    except Exception as e:
        print(f"Error during train-test split or scaling: {e}")
        return None

    try:
        if task_type == "classification":
            # Define classification models and parameters for GridSearchCV
            models = {
                'SVC': (SVC(), {'kernel': ['linear', 'rbf'], 'C': [1, 10], 'gamma': ['scale', 'auto']}),
                'RandomForest': (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [None, 10]}),
                'LogisticRegression': (LogisticRegression(), {'C': [1, 10]})
            }

            best_model = None
            best_params = None
            best_score = 0

            # Perform model selection and hyperparameter tuning for classification
            for model_name, (model, params) in models.items():
                grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
                grid_search.fit(X_train, y_train)
                if grid_search.best_score_ > best_score:
                    best_model = model_name
                    best_params = grid_search.best_params_
                    best_score = grid_search.best_score_

            print(f"Best Classification Model: {best_model}")
            print(f"Best Parameters: {best_params}")
            print(f"Best Cross-Validation Score: {best_score}")

            # Train the best model on the full training set
            final_model = models[best_model][0].set_params(**best_params)
            final_model.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = final_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            print("Classification Model Evaluation:")
            print(f"Accuracy: {accuracy}")
            print("Classification Report:")
            print(report)

            return {
                'selected_preprocessing': {
                    'imputed_columns': list(numeric_columns),
                    'encoded_columns': list(categorical_columns),
                    'scaling': True
                },
                'selected_model': best_model,
                'best_parameters': best_params,
                'test_accuracy': accuracy,
                'classification_report': report
            }

        elif task_type == "regression":
            # Define regression models and parameters for GridSearchCV
            models = {
                'SVR': (SVR(), {'kernel': ['linear', 'rbf'], 'C': [1, 10], 'gamma': ['scale', 'auto']}),
                'RandomForest': (RandomForestRegressor(), {'n_estimators': [50, 100], 'max_depth': [None, 10]}),
                'LinearRegression': (LinearRegression(), {})
            }

            best_model = None
            best_params = None
            best_score = -float('inf')

            # Perform model selection and hyperparameter tuning for regression
            for model_name, (model, params) in models.items():
                grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')  # Use MSE for regression
                grid_search.fit(X_train, y_train)
                if grid_search.best_score_ > best_score:
                    best_model = model_name
                    best_params = grid_search.best_params_
                    best_score = grid_search.best_score_

            print(f"Best Regression Model: {best_model}")
            print(f"Best Parameters: {best_params}")
            print(f"Best Cross-Validation Score (neg MSE): {best_score}")

            # Train the best model on the full training set
            final_model = models[best_model][0].set_params(**best_params)
            final_model.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = final_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print("Regression Model Evaluation:")
            print(f"Mean Squared Error: {mse}")
            print(f"R-squared: {r2}")

            return {
                'selected_preprocessing': {
                    'imputed_columns': list(numeric_columns),
                    'encoded_columns': list(categorical_columns),
                    'scaling': True
                },
                'selected_model': best_model,
                'best_parameters': best_params,
                'test_mse': mse,
                'test_r2': r2
            }

    except Exception as e:
        print(f"Error during model selection or evaluation: {e}")
        return None
