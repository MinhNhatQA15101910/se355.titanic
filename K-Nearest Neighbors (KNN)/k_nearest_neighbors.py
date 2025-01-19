# K-Nearest Neighbors (KNN)

## Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv("../train.csv")
X_train = dataset_train.drop(["Survived"], axis=1)
y_train = dataset_train["Survived"]
X_test = pd.read_csv("../test.csv")

numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Pclass", "Embarked"]

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

from sklearn.preprocessing import OneHotEncoder

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

## Define the model
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

## Create the complete pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

## Define hyperparameter grid for Grid Search
param_grid = {
    "classifier__n_neighbors": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ],
    "classifier__weights": ["uniform", "distance"],
    "classifier__metric": ["euclidean", "manhattan"],
}

## Grid search with 10-fold cross-validation
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(pipeline, param_grid, cv=10, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("Best parameters from Grid Search:", grid_search.best_params_)
print("Best score from Grid Search:", grid_search.best_score_)

## Evaluate the model using k-Fold Cross Validation
from sklearn.model_selection import cross_val_score

cross_val_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=10)
print("Cross-validation scores:", cross_val_scores)
print("Average cross-validation score:", np.mean(cross_val_scores))

## Make predictions using the best model from Grid Search
y_pred = grid_search.best_estimator_.predict(X_test)

## Export to csv
results = pd.DataFrame({"PassengerId": X_test["PassengerId"], "Survived": y_pred})
results.to_csv("knn_submission.csv", index=False)
