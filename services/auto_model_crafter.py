import os
from fastapi import UploadFile
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score


class AutoModelCrafter:
    def __init__(self, file: UploadFile):
        self.file = file
        self.file_path = None
        self.data = None
        self.numeric_columns = None
        self.categorical_columns = None
        self.label_encoders = {}
        self.task_type = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None

    async def save_file(self, upload_directory: str):
        """
        Saves the uploaded file to the given directory.
        """
        self.file_path = os.path.join(upload_directory, self.file.filename)
        try:
            with open(self.file_path, "wb") as buffer:
                buffer.write(await self.file.read())
            print(f"File saved to {self.file_path}")
        except Exception as e:
            raise Exception(f"Error saving file: {e}")

    def read_dataset(self):
        """
        Reads the dataset from the saved file. Supports CSV and JSON formats.
        """
        try:
            if self.file.filename.endswith('.csv'):
                self.data = pd.read_csv(self.file_path)
            else:
                self.data = pd.read_json(self.file_path)
            print("Dataset read successfully!")
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")
        return self.data

    def preprocess(self):
        """
        Preprocesses the dataset by imputing missing numeric values and encoding categorical features.
        Also separates features and target and determines whether the task is regression or classification.
        """
        try:
            # Handle missing numeric values.
            imputer = SimpleImputer(strategy='mean')
            self.numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
            self.categorical_columns = self.data.select_dtypes(include=['object']).columns

            if not self.numeric_columns.empty:
                self.data[self.numeric_columns] = imputer.fit_transform(self.data[self.numeric_columns])

            # Encode categorical columns.
            if not self.categorical_columns.empty:
                for col in self.categorical_columns:
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col].astype(str))
                    self.label_encoders[col] = le

            print("Preprocessing completed successfully!")
        except Exception as e:
            raise Exception(f"Error during preprocessing: {e}")

        # Separate features (X) and target (y)
        X = self.data.iloc[:, :-1]  # All columns except the last as features
        y = self.data.iloc[:, -1]   # Last column as the target variable

        print(f"Shape of X before split: {X.shape}")
        print(f"Shape of y before split: {y.shape}")

        # Determine task type based on the number of unique target values.
        if y.nunique() < 20:
            self.task_type = "classification"
            print("Classification task detected.")
        else:
            self.task_type = "regression"
            print("Regression task detected.")
        return X, y

    def split_and_scale(self, X, y):
        """
        Splits the dataset into training and testing sets and applies feature scaling.
        """
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            print("Train-test split and scaling completed successfully!")
        except Exception as e:
            raise Exception(f"Error during train-test split or scaling: {e}")

    def train_classification_model(self):
        """
        Performs model selection and hyperparameter tuning for classification tasks,
        then trains and evaluates the best model.
        """
        models = {
            'SVC': (SVC(), {'kernel': ['linear', 'rbf'], 'C': [1, 10], 'gamma': ['scale', 'auto']}),
            'RandomForest': (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [None, 10]}),
            'LogisticRegression': (LogisticRegression(max_iter=1000), {'C': [1, 10]})
        }

        best_model = None
        best_params = None
        best_score = 0

        for model_name, (model, params) in models.items():
            grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
            grid_search.fit(self.X_train, self.y_train)
            if grid_search.best_score_ > best_score:
                best_model = model_name
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_

        print(f"Best Classification Model: {best_model}")
        print(f"Best Parameters: {best_params}")
        print(f"Best Cross-Validation Score: {best_score}")

        # Train the best model on the full training set.
        final_model = models[best_model][0].set_params(**best_params)
        final_model.fit(self.X_train, self.y_train)

        # Evaluate the model on the test set.
        y_pred = final_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        print("Classification Model Evaluation:")
        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(report)

        return {
            'selected_preprocessing': {
                'imputed_columns': list(self.numeric_columns),
                'encoded_columns': list(self.categorical_columns),
                'scaling': True
            },
            'selected_model': best_model,
            'best_parameters': best_params,
            'test_accuracy': accuracy,
            'classification_report': report
        }

    def train_regression_model(self):
        """
        Performs model selection and hyperparameter tuning for regression tasks,
        then trains and evaluates the best model.
        """
        models = {
            'SVR': (SVR(), {'kernel': ['linear', 'rbf'], 'C': [1, 10], 'gamma': ['scale', 'auto']}),
            'RandomForest': (RandomForestRegressor(), {'n_estimators': [50, 100], 'max_depth': [None, 10]}),
            'LinearRegression': (LinearRegression(), {})
        }

        best_model = None
        best_params = None
        best_score = -float('inf')

        for model_name, (model, params) in models.items():
            grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(self.X_train, self.y_train)
            if grid_search.best_score_ > best_score:
                best_model = model_name
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_

        print(f"Best Regression Model: {best_model}")
        print(f"Best Parameters: {best_params}")
        print(f"Best Cross-Validation Score (neg MSE): {best_score}")

        # Train the best model on the full training set.
        final_model = models[best_model][0].set_params(**best_params)
        final_model.fit(self.X_train, self.y_train)

        # Evaluate the model on the test set.
        y_pred = final_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print("Regression Model Evaluation:")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")

        return {
            'selected_preprocessing': {
                'imputed_columns': list(self.numeric_columns),
                'encoded_columns': list(self.categorical_columns),
                'scaling': True
            },
            'selected_model': best_model,
            'best_parameters': best_params,
            'test_mse': mse,
            'test_r2': r2
        }

    async def craft_model(self, upload_directory: str):
        """
        Runs the full auto-ML pipeline:
          1. Save the uploaded file.
          2. Read the dataset.
          3. Preprocess the data.
          4. Split and scale the data.
          5. Select, train, and evaluate the best model.
        Returns the evaluation results.
        """
        await self.save_file(upload_directory)
        self.read_dataset()
        X, y = self.preprocess()
        self.split_and_scale(X, y)

        if self.task_type == "classification":
            return self.train_classification_model()
        elif self.task_type == "regression":
            return self.train_regression_model()
        else:
            raise Exception("Unknown task type encountered during preprocessing.")
