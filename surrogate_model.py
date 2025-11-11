import ConfigSpace

import sklearn.impute
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd


class SurrogateModel:

    def __init__(self, config_space):
        self.config_space = config_space
        self.df = None
        self.model = None

    def fit(self, df):
        """
        Receives a data frame, in which each column (except for the last two) represents a hyperparameter, the
        penultimate column represents the anchor size, and the final column represents the performance.

        :param df: the dataframe with performances
        :return: Does not return anything, but stores the trained model in self.model
        """
        self.df = df
        X = df.drop('score', axis=1)
        y = df['score']

        #Since feature columns are both categorical as numerical we can need different
        #data pre-processing steps for each
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
        
        #For the numerical values
        numerical_transformer = Pipeline(steps=[
            ('imputer', sklearn.impute.SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        #For the categorical values
        categorical_transformer = Pipeline(steps=[
            ('imputer', sklearn.impute.SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        #So we can combine both pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        #Combine the steps into one PIPELINE
        self.model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor())])
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        self.model.fit(X_train, y_train)
                
        #Small test script to eval
        y_pred = self.model.predict(X_test)
        spearman_corr, _ = spearmanr(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Model Performance:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R-squared (R2): {r2:.4f}")
        print(f"Spearman Correlation: {spearman_corr:.4f}")

    def predict(self, theta_new, anchor=None):
        """
        Predicts the performance of a given configuration theta_new

        :param theta_new: a dict, where each key represents the hyperparameter (or anchor)
        :return: float, the predicted performance of theta new (which can be considered the ground truth)
        """

        #Some hyperparameters are not always included in theta_new
        #Therefore, we first initialize a df with the known columns
        if anchor is not None:
            theta_new['anchor_size'] = anchor

        original_columns = self.df.drop('score', axis=1).columns
        X = pd.DataFrame(columns=original_columns)
        X.loc[0] = {k: v for k, v in theta_new.items() if k in original_columns}

        return self.model.predict(X).item()