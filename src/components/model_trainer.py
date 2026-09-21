import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:,-1],
                test_array[:, :-1],
                test_array[:,-1]
            )
            # Model List for training
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False, thread_count = 2)
            }

            # Hyperparameter Grids for the above algorithms
            params = {
                "Linear Regression": {},
                "Lasso": {
                    'alpha': [0.1, 1.0, 10.0]
                },
                "Ridge": {
                    'alpha': [0.1, 1.0, 10.0]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [64, 128, 256]
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9]
                },
                "Decision Tree Regressor": {
                    'criterion': ['squared_error', 'absolute_error']
                },
                "Random Forest Regressor": {
                    'n_estimators': [64, 128, 256]
                },
                "XGBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [64, 128, 256]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 1.0],
                    'n_estimators': [50, 100, 200]
                },
                "CatBoost Regressor": {
                    # 'depth':,
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                }
            }

            model_report = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models = models, params = params)

            model_report = model_report.sort_values(by="test_model_score", ascending=False)
            # To get the best model score from the model report
            best_model_score = model_report["test_model_score"].iloc[0]
            best_model_name = model_report["model_name"].iloc[0]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)