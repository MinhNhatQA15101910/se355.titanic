import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv("train.csv")
X_train = dataset_train.drop(["Survived"], axis=1)
y_train = dataset_train["Survived"]
X_test = pd.read_csv("test.csv")

numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
label_categorical_features = ["Sex"]
onehot_categorical_features = ["Embarked"]

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

label_categorical_transformer = Pipeline(steps=[("label", LabelEncoder())])

onehot_categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("label_cat", label_categorical_transformer, label_categorical_features),
        ("onehot_cat", onehot_categorical_transformer, onehot_categorical_features),
    ]
)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print(y_pred)

results = pd.DataFrame({"PassengerId": X_test["PassengerId"], "Survived": y_pred})
results.to_csv("knn_submission.csv", index=False)
